import math
import torch
import torch.backends.cudnn as cudnn
from bias_transfer.trainer import apply_noise, load_checkpoint
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib import cm
from sklearn.cluster import AgglomerativeClustering
from nnfabrik.main import *
from bias_transfer.tables.base import *
# from bias_transfer.tables.transfer import *

import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Analyzer:

    def __init__(self,
                 tables=TrainedModel(),
                 selection_like=None,
                 selection_exact=None,
                 reformat_comment=(),
                 comment_map={}):
        self.df = self._reformat_comments(self._load_data(tables, selection_like, selection_exact), reformat_comment)
        self.df = self._remap_comments(self.df, comment_map)

    def _load_data(self, tables, selection_like, selection_exact):
        # Select data:
        fetched = []
        for table in tables:
            if selection_exact:
                fetched += (table & selection_exact).fetch("output", *selection_exact[0].keys(), as_dict=True)
            elif selection_like:
                selection_list = []
                for k, v in selection_like.items():
                    if not isinstance(v, tuple):
                        v = (v,)
                    selection_list.append("{} LIKE '%{}%'".format(k, "%".join(v)))
                print(selection_list)
                dj.AndList(selection_list)
                selection_keys = set("comment")
                for k in selection_like.keys():
                    if k.startswith("NOT"):
                        selection_keys.add(k[4:])
                    else:
                        selection_keys.add(k)
                fetched += (table & selection_list).fetch("output", *selection_keys, as_dict=True)
            else:
                fetched += table.fetch("output", "comment", as_dict=True)


        # Reformat to display nicely/ access easily:
        df = pd.DataFrame(fetched)
        df = pd.concat([df.drop(['output'], axis=1), df['output'].apply(pd.Series)], axis=1)
        df = df.rename(columns={0: "training_progress"})
        df = pd.concat([df.drop([1], axis=1), df[1].apply(pd.Series)], axis=1)  # Final Results
        return df

    def _remap_comments(self, df, comment_map):
        # reformat comment
        def remap(c):
            for k,v in comment_map.items():
                if isinstance(k,tuple):
                    match = True
                    for sub_k in k:
                        if sub_k not in c:
                            match = False
                    if match:
                        return v
                else:
                    if k == c:
                        return v
                    elif c.startswith(k):
                        return v
                    elif k in c:
                        return v
            print(c)
            print("********")
            return c
        df["comment"] = df["comment"].map(remap)
        return df

    def _reformat_comments(self, df, reformat_comment):
        # reformat comment
        def reformat(comment):
            for remove in reformat_comment:
                s_pos = comment.find(remove)
                if s_pos == -1:
                    continue
                comment = comment[:s_pos] + comment[s_pos + len(remove):]
            return comment

        df["comment"] = df["comment"].map(reformat)
        if "transfer_comment" in df.columns:
            df["transfer_comment"] = df["transfer_comment"].map(reformat)
            df["comment"] = df["comment"] + "\nTRANSFER." + df["transfer_comment"]
        df["comment"] = df["comment"].map(lambda x: "default" if x == "" else x)
        df = df.sort_values(by=['comment'])
        return df

    def plot(self,
             to_plot="dev_noise_acc",
             noise_measure="std",
             plot_method=sns.catplot,
             kind="bar",
             save="",
             fig=None, ax=None):
        # import matplotlib as mpl
        # from matplotlib import pyplot as plt
        # font = {'family': 'Helvetica Neue',
        #         'weight': 'bold',
        #         'size': 22}
        # mpl.rc('font', **font)
        if not fig or not ax:
            self.fs = (15, 10)
            self.dpi = 200
            sns.set()
            sns.set_context("talk")
            plt.style.use("dark_background")
            # sns.set_palette("pastel")
            fig, ax = plt.subplots(figsize=self.fs, dpi=self.dpi)
        # Plot
        if to_plot in ("test_acc", "test_loss"):
            plot_method(x="comment", y=to_plot, hue="comment",
                        kind="bar", data=self.df, ax=ax)
        if to_plot in ("dev_noise_acc", "dev_noise_loss"):
            data = self.df[to_plot].apply(pd.Series)
            data = data['noise_' + noise_measure].apply(pd.Series)
            data = pd.concat([self.df["comment"], data], axis=1)
            data.index = data.comment
            del data["comment"]
            data = data.stack().reset_index()
            data.columns = ["Training", 'Noise in Evaluation (Standard deviation)', 'Accuracy']
            data["Noise in Evaluation (Standard deviation)"] = data["Noise in Evaluation (Standard deviation)"].map(
                lambda x: float(x.split("_")[0]))
            if kind:
                plot_method(data=data, x="Noise in Evaluation (Standard deviation)", y="Accuracy", hue="Training",
                            kind=kind, ax=ax)
            else:
                plot_method(data=data, x="Noise in Evaluation (Standard deviation)", y="Accuracy", hue="Training",
                            ax=ax)
        if to_plot in ("training_progress",):
            data = self.df[to_plot].apply(pd.Series)
            data = data.applymap(lambda x: x.get('dev_acc'))
            data = pd.concat([self.df["comment"], data], axis=1)
            data.index = data.comment
            del data["comment"]
            data = data.stack().reset_index()
            data.columns = ["comment", 'epoch', 'score']
            plot_method(x="epoch", y="score", hue="comment", data=data, ax=ax)

        sns.despine(offset=10, trim=True)
        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        if save:
            fig.savefig(save, facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor(), bbox_inches='tight')
        return fig, ax


def print_table_for_excel(table):
    prior_columns = 1
    keys = []
    for key in table.fetch(as_dict=True)[0].keys():
        if "comment" in key:
            keys.append(key)
        if key == "transfer_trainer_hash" or key == "transfer_trainer_config":
            keys.append(key)
            prior_columns = 3
    # heading
    row = table.fetch("output", as_dict=True)[0]["output"][1]["dev_noise_acc"]
    print("," * prior_columns, end="")
    for key in row.keys():
        print(key + ("," * (len(row[key]))), end="")
    print()
    print("," * prior_columns, end="")
    for key in row.keys():
        for sub_key in row[key].keys():
            print(sub_key.split("_")[0], end=",")
    print()
    # content
    for row in table.fetch("output", *keys, as_dict=True):
        comment = []
        extra = []
        for k, v in row.items():
            if k == "output" or "hash" in k:
                continue
            elif "config" in k:
                extra = ["freeze_{}".format(v['freeze']), "reset_{}".format(v['reset_linear'])]
            else:
                comment.append(v)
        print(".".join(comment), end=", ")
        if extra:
            print(",".join(extra), end=", ")
        output = row["output"]
        final_results = output[1]['dev_noise_acc']
        for key in final_results.keys():
            for res in final_results[key].items():
                print(res[1], end=",")
        print()


def print_trained_transfer():
    results = (TrainedTransferModel()
               * Trainer.proj(transfer_trainer_hash="trainer_hash", transfer_trainer_fn="trainer_fn",
                              transfer_trainer_comment="trainer_comment", transfer_trainer_config="trainer_config")
               * Model.proj(transfer_model_hash="model_hash", transfer_model_fn="model_fn",
                            transfer_model_comment="model_comment")
               * Dataset.proj(transfer_dataset_hash="dataset_hash", transfer_dataset_fn="dataset_fn",
                              transfer_dataset_comment="dataset_comment")
               )
    print_table_for_excel(results)


def print_trained():
    print_table_for_excel(TrainedModel())


def plot_training_dev(col_labels, pretrained_labels, pretrained_output, transfer_labels, transfer_output, max_x=2,
                      category="loss"):
    max_y = math.ceil((len(pretrained_labels)) / max_x)
    fig, ax = plt.subplots(max_y, max_x, sharex=True, sharey=True, figsize=(8, 10))
    if max_y == 1:
        ax = [ax]
    if max_x == 1:
        for i, a in enumerate(ax):
            ax[i] = [a]
    fig.set_dpi(200)
    width = 0.8
    fig.tight_layout()
    for i, pretrained_model in enumerate(pretrained_labels):
        x = i % max_x
        y = i // max_x
        comment = pretrained_model["comment"]
        s_pos = comment.find(".noise_") + len(".noise_")
        e_pos = comment.find("_", s_pos + 4)
        ax[y][x].set_title(comment, fontsize=5)
        matches = []
        for t, l in enumerate(transfer_labels):
            if l["comment"] == comment:
                matches.append(t)
        total = len(matches) + 1
        for key, D in pretrained_output[i].items():
            if category not in key:
                continue
            #             print("pretrain:", len(D))
            ax[y][x].plot(range(len(D)), D, label='{} Only Noisy'.format(key))
        for j, t in enumerate(matches):
            for key, D in transfer_output[t].items():
                #                 print("transfer:", len(D))
                if category not in key:
                    continue
                ax[y][x].plot(range(len(D)), D,
                              label='{} Transfer with reset_linear={}'.format(key, transfer_labels[t]["reset_linear"]))
        ax[y][x].grid()

        ax[y][x].legend(fontsize=5)
    fig.show()


def plot_noise_eval(col_labels, pretrained_labels, pretrained_output, transfer_labels, transfer_output,
                    max_x=3, save='', restrict=()):
    max_y = math.ceil((len(pretrained_labels)) / max_x)
    if restrict:
        max_y = math.ceil((len(restrict)) / max_x)
    fig, ax = plt.subplots(max_y, max_x, sharex=True, sharey=True, figsize=(8, 4))
    if max_y == 1:
        ax = [ax]
    if max_x == 1:
        for i, a in enumerate(ax):
            ax[i] = [a]
    fig.set_dpi(200)
    width = 0.8
    fig.tight_layout()
    i = 0
    for p, pretrained_model in enumerate(pretrained_labels):
        comment = pretrained_model["comment"]
        s_pos = comment.find(".noise_") + len(".noise_")
        e_pos = comment.find("_", s_pos + 4)
        title = comment
        # title = comment[s_pos:e_pos]
        if restrict and title not in restrict:
            continue
        x = i % max_x
        y = i // max_x
        ax[y][x].set_title(title, fontsize=5)
        matches = []
        for t, l in enumerate(transfer_labels):
            if l["comment"] == comment:
                matches.append(t)
        total = len(matches) + 1
        D = pretrained_output[p]
        pos = np.array(range(len(D))) - total // 2 * width / total
        ax[y][x].bar(pos, D, width=width / total, label='Pretrain')
        for j, t in enumerate(matches):
            D = transfer_output[t]
            ax[y][x].bar(pos + (j + 1) * (width / total), D, width=width / total,
                         label='Transfer with reset_linear={}'.format(transfer_labels[t]["reset_linear"]))
        i += 1
    for x in range(max_x):
        for y in range(max_y):
            ax[y][x].set_xticks(pos + 0.5 * width)
            ax[y][x].set_xticklabels(col_labels, fontsize=6, rotation='vertical')
            ax[y][x].set_yticks(np.arange(0, 110, 10))
            ax[y][x].yaxis.grid()

            ax[y][x].legend(fontsize=5)
    fig.show()
    if save:
        fig.savefig(save)


def extract_results(dj_table, result_category: str = 'dev_noise_acc'):
    chkpts = result_category.startswith("chkpts")
    col_labels = []
    if not chkpts:
        # heading
        row = dj_table.fetch("output", as_dict=True)[0]["output"][1][result_category]
        for key in row.keys():
            for sub_key in row[key].keys():
                col_labels.append(key.split("_")[1] + "_" + sub_key.split("_")[0])
    # content
    labels = []
    results = []
    for row in dj_table.fetch(as_dict=True):
        row_labels = {}
        for k, v in row.items():
            if "config" in k:
                row_labels['freeze'] = v['freeze']
                row_labels['reset_linear'] = v['reset_linear']
            elif "comment" in k:
                row_labels[k] = v
        labels.append(row_labels)
        row = row["output"][0] if result_category.startswith("chkpts") else row["output"][1][result_category].values()
        row_res = {k: [] for k in row[0].keys()} if chkpts else []
        for res_coll in row:
            if chkpts:
                for key, res in res_coll.items():
                    row_res[key].append(res)
            else:
                for res in res_coll.values():
                    row_res.append(res)
        results.append(row_res)
    return col_labels, labels, results


def visualize_data(data_loader,
                   num_samples: int = 2,
                   add_noise: bool = False,
                   noise_stds: tuple = (None,),
                   noise_snrs: tuple = (None,),
                   force_cpu: bool = False,
                   comment: str = ""):
    print('==> Starting visualization {}'.format(comment), flush=True)
    device = 'cuda' if torch.cuda.is_available() and not force_cpu else 'cpu'
    fig, axs = plt.subplots(num_samples, len(noise_stds) + len(noise_snrs), sharex=True, sharey=True)
    fig.set_dpi(200)
    fig.tight_layout()  # Or equivalently,  "plt.tight_layout()"
    with torch.no_grad():
        if add_noise:
            torch.manual_seed(42)  # so that we always have the same noise for evaluation!
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            if batch_idx == num_samples:
                return
            for i, noise_std in enumerate(noise_stds):
                if add_noise and noise_std:
                    inputs_, _ = apply_noise(inputs.clone(), device, std=noise_std, snr=None)
                else:
                    inputs_ = inputs.clone()
                inputs_ = inputs_.transpose(1, 3)
                axs[batch_idx, i].imshow(inputs_[0, :].cpu().numpy())
                if batch_idx == 0:
                    axs[batch_idx, i].set_title('{}'.format(list(noise_std.keys())[0]))
            for i, noise_snr in enumerate(noise_snrs):
                if add_noise and noise_snr:
                    inputs_ , _ = apply_noise(inputs.clone(), device, std=None, snr=noise_snr)
                else:
                    inputs_ = inputs.clone()
                inputs_ = inputs_.transpose(1, 3)
                axs[batch_idx, i + len(noise_stds)].imshow(inputs_[0, :].cpu().numpy())
                if batch_idx == 0:
                    axs[batch_idx, i + len(noise_stds)].set_title('{}'.format(list(noise_snr.keys())[0]))
    return fig


def visualize_corr_matrix(model, data_loader,
                          num_samples: int = 2,
                          layer: int = 0,
                          add_noise: bool = False,
                          noise_std: float = None,
                          noise_snr: float = 0.9,
                          force_cpu: bool = False,
                          load_from_path: str = "",
                          comment: str = ""):
    print('==> Starting visualization {}'.format(comment), flush=True)
    device = 'cuda' if torch.cuda.is_available() and not force_cpu else 'cpu'
    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    if load_from_path:
        model, _, _ = load_checkpoint(load_from_path, model)
    fig, axs = plt.subplots(num_samples, sharex=True, sharey=True)
    fig.set_dpi(200)
    fig.tight_layout()  # Or equivalently,  "plt.tight_layout()"

    model.eval()
    with torch.no_grad():
        if add_noise:
            torch.manual_seed(42)  # so that we always have the same noise for evaluation!
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            if batch_idx == num_samples:
                return
            inputs, targets = inputs.to(device), targets.to(device)
            if add_noise:
                inputs = apply_noise(inputs, device, std=noise_std, snr=noise_snr)
            _, mat = model(inputs, compute_corr=True)

            # clusters = [AgglomerativeClustering(n_clusters=12).fit(torch.sqrt(1 - mat[i].detach().cpu())) for i
            clusters = [AgglomerativeClustering(n_clusters=12).fit(1 - mat[i].detach().cpu()) for i
                        in range(0, len(mat))]
            indices = [np.argsort(cluster.labels_) for cluster in clusters]
            axs[batch_idx].matshow(mat[layer][indices[layer]][:, indices[layer]].detach().cpu(),
                                   cmap=cm.get_cmap(name="Spectral_r"))
            axs[batch_idx].matshow(mat[layer].detach().cpu(), cmap=cm.get_cmap(name="Spectral_r"))
            # f.suptitle("Layer " + str(i) + " Batch-wise Correlation Similarity Matrices", fontsize=20)
    return fig

