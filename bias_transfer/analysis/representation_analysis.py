import copy

import torch

import bias_transfer.trainer.main_loop
import bias_transfer.trainer.trainer
from nnfabrik.utility.dj_helpers import make_hash
from sklearn.cluster import AgglomerativeClustering
from torch import nn
from bias_transfer import trainer
from bias_transfer.trainer.main_loop_modules.noise_augmentation import NoiseAugmentation
from torch.backends import cudnn
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import os


class RepresentationAnalyser:
    def __init__(
        self,
        experiment,
        table,
        dataset: str = "val",
        path: str = "/work/analysis/",
        plot_style: str = "lightpaper",
        rep_name: str = "conv_rep",
    ):
        self.experiment = experiment
        self.dataset = dataset
        data_loaders, self.model, self.trainer = (
            table & experiment.get_restrictions()
        ).load_model(include_trainer=True, include_state_dict=True, seed=42)
        self.num_samples = -1
        self.sample_loader = torch.utils.data.DataLoader(
            data_loaders[dataset].dataset,
            sampler=data_loaders[dataset].sampler,
            batch_size=64,
            shuffle=False,
            num_workers=1,
            pin_memory=False,
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self._reset_seed()
        self.criterion = nn.CrossEntropyLoss()
        self.clean_rep, self.noisy_rep, self.clean_df, self.noisy_df = (
            None,
            {},
            None,
            {},
        )
        self.feat_cols = None
        self.path = path
        self.style = plot_style
        self.rep_name = rep_name

    def _reset_seed(self):
        torch.manual_seed(42)
        np.random.seed(42)
        if self.device == "cuda":
            cudnn.benchmark = False
            cudnn.deterministic = True
            torch.cuda.manual_seed(42)

    def _plot_preparation(self, nrows=1, ncols=1):
        fs = (16, 10) if "talk" in self.style else (12, 7.5)
        dpi = 200 if "talk" in self.style else 200
        sns.set()
        if "light" in self.style:
            sns.set_style("whitegrid")
        if "ticks" in self.style:
            sns.set_style("ticks")
        if "dark" in self.style:
            plt.style.use("dark_background")
        if "talk" in self.style:
            sns.set_context("talk")
        else:
            sns.set_context("paper")
        fig, ax = plt.subplots(nrows, ncols, figsize=fs, dpi=dpi)
        return fig, ax

    def _compute_representation(self, main_loop_modules):
        (
            acc,
            loss,
            module_losses,
            collected_outputs,
        ) = bias_transfer.trainer.main_loop.main_loop(
            self.model,
            self.criterion,
            self.device,
            None,
            self.sample_loader,
            0,
            main_loop_modules,
            train_mode=False,
            return_outputs=True,
        )
        outputs = [o[self.rep_name] for o in collected_outputs]
        print("Acc:", acc, "Loss:", loss, flush=True)
        return torch.cat(outputs), acc

    def _generate_gif(self, filenames, name):
        import imageio

        images = []
        for filename in filenames:
            images.append(imageio.imread(filename))
        imageio.mimsave(
            os.path.join(self.path, "{}.gif".format(name)),
            images,
            format="GIF",
            duration=2,
        )

    def run(self, method):
        if method in ("pca", "tsne"):
            to_run = self.dim_reduction
        else:
            to_run = self.corr_matrix
        filenames = []
        clean_rep = to_run(noise_level=0.0, method=method, mode="clean")
        filenames.append(
            os.path.join(self.path, self._get_name(method, "clean", 0.0) + "_plot.png")
        )
        for i in range(1, 21):
            noise_level = 0.05 * i
            to_run(
                noise_level=noise_level,
                method=method,
                mode="noisy",
                clean_rep=clean_rep,
            )
            filenames.append(
                os.path.join(
                    self.path,
                    self._get_name(method, "noisy", noise_level) + "_plot.png",
                )
            )
        self._generate_gif(filenames, self._get_name(method=method))

    def clean_vs_noisy(self, noise_level=0.0):
        print("==> Computing Representations", flush=True)
        self._reset_seed()
        if self.clean_rep is None:
            # Representations form clean data:
            print("Compute representation of clean input", flush=True)
            self.clean_rep = self._compute_representation([])
        else:
            print("Representation of clean input already in memory")

        # Representations from noisy data:
        print("Compute representation of noisy input", flush=True)
        self._reset_seed()
        experiment = copy.deepcopy(self.experiment)
        bias_transfer.trainer.trainer.trainer.noise_std = {noise_level: 1.0}
        main_loop_modules = [
            NoiseAugmentation(
                config=bias_transfer.trainer.trainer.trainer,
                device=self.device,
                data_loader=self.sample_loader,
                seed=42,
            )
        ]
        self.noisy_rep = self._compute_representation(main_loop_modules)

    def _cosine_loss(self, rep_1, rep_2):
        # Compare
        cosine_criterion = nn.CosineEmbeddingLoss()
        return cosine_criterion(
            rep_1, rep_2, torch.ones(rep_1.shape[:1], device=self.device)
        )

    def _mse_loss(self, rep_1, rep_2):
        mse_criterion = nn.MSELoss()
        return mse_criterion(rep_1, rep_2)

    def clean_vs_noisy_distance(self, noise_level=0.0):
        self.clean_vs_noisy(noise_level)
        cosine = self._cosine_loss(self.clean_rep[0], self.noisy_rep[0])
        mse = self._mse_loss(self.clean_rep[0], self.noisy_rep[0])
        print(
            "Clean vs. Noisy: Cosine loss:",
            cosine.item(),
            "MSE loss:",
            mse.item(),
            flush=True,
        )

    def _convert_to_df(self, rep, noise_level=0.0):
        torch.manual_seed(42)
        np.random.seed(42)
        if self.device == "cuda":
            torch.cuda.manual_seed(42)
        rep = rep.cpu()
        targets = torch.cat([t for _, t in self.sample_loader]).cpu()
        self.num_labels = max(targets) + 1
        feat_cols = ["dim" + str(i) for i in range(rep.shape[1])]
        df = pd.DataFrame(rep, columns=feat_cols)
        df["y"] = targets
        df["label"] = df["y"].apply(lambda i: str(i))
        df["noise"] = np.ones_like(targets) * noise_level
        if self.num_samples > 0:
            # For reproducability of the results
            np.random.seed(42)
            rndperm = np.random.permutation(df.shape[0])
            df = df.loc[rndperm[: self.num_samples], :].copy()
        return df, feat_cols

    def _clean_vs_noisy_df(self, noise_level=0.0):
        self.clean_vs_noisy(noise_level=noise_level)
        if self.clean_df is None:
            self.clean_df, self.feat_cols = self._convert_to_df(self.clean_rep[0], 0.0)
        self.noisy_df, _ = self._convert_to_df(self.noisy_rep[0], noise_level)

    def _save_representation(self, to_save, method, mode, noise_level):
        name = self._get_name(method, mode, noise_level) + ".npy"
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        np.save(os.path.join(self.path, name), to_save)

    def _load_representation(self, method, mode, noise_level):
        name = self._get_name(method, mode, noise_level) + ".npy"
        file = os.path.join(self.path, name)
        if os.path.isfile(file):
            print("Found existing {} result that will be loaded now".format(method))
            return np.load(file)
        return None

    def _compute_pca(self, df, mode, noise_level, pca=None):
        pca_result = self._load_representation("pca", mode, noise_level)
        if pca_result is None:
            if not pca:
                pca = PCA(n_components=3)
                pca.fit(df[self.feat_cols].values)
            pca_result = pca.transform(df[self.feat_cols].values)
            self._save_representation(pca_result, "pca", mode, noise_level)
            print(
                "Explained variation per principal component: {}".format(
                    pca.explained_variance_ratio_
                ),
                flush=True,
            )
        df["pca-one"] = pca_result[:, 0]
        df["pca-two"] = pca_result[:, 1]
        df["pca-three"] = pca_result[:, 2]
        return pca

    def _compute_tsne(self, df, mode, noise_level):
        tsne_result = self._load_representation("tsne", mode, noise_level)
        if tsne_result is None:
            tsne = TSNE(
                n_components=2, verbose=1, perplexity=40, n_iter=250, init="pca"
            )
            tsne_result = tsne.fit_transform(df[self.feat_cols].values)
            self._save_representation(tsne_result, "tsne", mode, noise_level)
        df["tsne-2d-one"] = tsne_result[:, 0]
        df["tsne-2d-two"] = tsne_result[:, 1]

    def _plot_dim_reduction(
        self,
        df,
        data_columns,
        num_labels=100,
        hue="y",
        style=None,
        title="",
        file_name="",
        legend=False,
        acc=None,
    ):
        fig, ax = self._plot_preparation(1, len(data_columns))
        if not isinstance(ax, list):
            ax = [ax]
        for i, (x, y) in enumerate(data_columns):
            sns.scatterplot(
                x=x,
                y=y,
                hue=hue,
                style=style,
                palette=sns.color_palette("hls", num_labels),
                data=df,
                legend=legend,
                s=10,
                # ec=None,
                ax=ax[i],
            )
            if acc:
                ax[i].text(
                    0.85,
                    0.90,
                    "Accuracy: {:02.2f}".format(acc),
                    transform=ax[i].transAxes,
                )
        sns.despine(offset=10, trim=True)
        if title:
            fig.suptitle(title, fontsize=16)
        if file_name:
            fig.savefig(
                os.path.join(self.path, file_name),
                facecolor=fig.get_facecolor(),
                edgecolor=fig.get_edgecolor(),
                bbox_inches="tight",
            )
            plt.close(fig)

    def _get_name(self, method, mode=None, noise_level=None):
        if method and mode and noise_level:
            return "_".join(
                [
                    make_hash(self.experiment.get_key()),
                    self.rep_name,
                    method,
                    mode,
                    self.dataset,
                    str(int(noise_level * 100)),
                ]
            )
        else:
            return "_".join(
                [
                    make_hash(self.experiment.get_key()),
                    self.rep_name,
                    method,
                    self.dataset,
                ]
            )

    def dim_reduction(
        self, method="tsne", mode="combined", noise_level=0.0, clean_rep=None
    ):
        self._clean_vs_noisy_df(noise_level=noise_level)
        if mode == "combined":
            combined_df = pd.DataFrame(self.clean_df)
            combined_df = combined_df.append(self.noisy_df, ignore_index=True)
            df = combined_df
            acc = self.noisy_rep[1]
            title = "Rep noisy vs clean data "
        elif mode == "noisy":
            title = "Rep from noisy data (std = {:01.2f})".format(noise_level)
            df = self.noisy_df
            acc = self.noisy_rep[1]
        else:
            title = "Rep from clean data "
            df = self.clean_df
            acc = self.clean_rep[1]

        data_columns = []
        print("==> Computing {} representation".format(method))
        if "tsne" in method:
            self._compute_tsne(df, mode, noise_level)
            data_columns.append(("tsne-2d-one", "tsne-2d-two"))
        if "pca" in method:
            clean_rep = self._compute_pca(df, mode, noise_level, pca=clean_rep)
            data_columns.append(("pca-one", "pca-two"))

        print("==> Plotting {} representation".format(method))
        self._plot_dim_reduction(
            df,
            data_columns,
            num_labels=self.num_labels,
            style="noise" if "combined" in mode else None,
            hue="y",
            title=title + "\n" + "Model: " + self.experiment.comment,
            file_name=self._get_name(method, mode, noise_level) + "_plot",
            acc=acc,
        )
        return clean_rep

    def _plot_corr_matrix(
        self, mat, title="", file_name="", n_clusters=10, indices=None, acc=None
    ):
        fig, ax = self._plot_preparation(1, 1)
        if indices is None:
            clusters = AgglomerativeClustering(n_clusters=n_clusters).fit(1 - mat)
            indices = np.argsort(clusters.labels_)
        sns.heatmap(
            mat[indices][:, indices],
            cmap="YlGnBu",
            xticklabels=400,
            yticklabels=400,
            vmin=0.0,
            vmax=1.0,
        )
        # sns.heatmap(mat[indices][:, indices], cmap="YlGnBu", xticklabels=400, yticklabels=400)
        sns.despine(offset=10, trim=True)
        if title:
            fig.suptitle(title, fontsize=16)
        if acc:
            ax.text(
                0.82, 0.93, "Accuracy: {:02.2f}".format(acc), transform=ax.transAxes
            )
        if file_name:
            fig.savefig(
                os.path.join(self.path, file_name),
                facecolor=fig.get_facecolor(),
                edgecolor=fig.get_edgecolor(),
                bbox_inches="tight",
            )
            plt.close(fig)
        return indices

    def _compute_corr_matrix(self, x, mode, noise_level):
        result = self._load_representation("corr", mode, noise_level)
        if result is None:
            x_flat = x.flatten(1, -1)
            # centered = (x_flat - x_flat.mean()) / x_flat.std()
            # result = (centered @ centered.transpose(0, 1)) / x_flat.size()[1]
            centered = x_flat - x_flat.mean(dim=1).view(-1, 1)
            result = (centered @ centered.transpose(0, 1)) / torch.ger(
                torch.norm(centered, 2, dim=1), torch.norm(centered, 2, dim=1)
            )  # see https://de.mathworks.com/help/images/ref/corr2.html
            print(torch.max(result))
            result = result.detach().cpu()
            self._save_representation(result, "corr", mode, noise_level)
        return result

    def corr_matrix(
        self, mode="clean", noise_level=0.0, clean_rep=None, *args, **kwargs
    ):
        self.clean_vs_noisy(noise_level=noise_level)
        title = "Correlation matrix for rep from {} data ".format(mode)
        if mode == "noisy":
            corr_matrix = self._compute_corr_matrix(
                self.noisy_rep[0], mode, noise_level
            )
            title += "(std = {:01.2f})".format(noise_level)
            acc = self.noisy_rep[1]
        else:
            corr_matrix = self._compute_corr_matrix(self.clean_rep[0], mode, 0.0)
            acc = self.clean_rep[1]

        clean_rep = self._plot_corr_matrix(
            corr_matrix,
            title=title + "\n" + "Model: " + self.experiment.comment,
            file_name=self._get_name("corr", mode, noise_level) + "_plot",
            indices=clean_rep,
            acc=acc,
        )
        return clean_rep
