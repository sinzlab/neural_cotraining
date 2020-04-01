import torch
from nnfabrik.utility.dj_helpers import make_hash
from torch import nn
from bias_transfer.trainer import main_loop
from bias_transfer.trainer.noise_augmentation import NoiseAugmentation
from torch.backends import cudnn
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


def compute_representation(model, criterion, device, sample_loader, main_loop_modules):
    acc, loss, module_losses, collected_outputs = main_loop(model, criterion, device, None, sample_loader, 0,
                                                            main_loop_modules, train_mode=False, return_outputs=True)
    outputs = [o["conv_rep"] for o in collected_outputs]
    print("Acc:", acc, "Loss:", loss, flush=True)
    return torch.cat(outputs)


def compute_pca(df_subset, data_subset):
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(data_subset)
    df_subset['pca-one'] = pca_result[:, 0]
    df_subset['pca-two'] = pca_result[:, 1]
    df_subset['pca-three'] = pca_result[:, 2]
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_), flush=True)


def compute_tsne(df_subset, data_subset):
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=250, init='pca')
    tsne_results = tsne.fit_transform(data_subset)

    df_subset['tsne-2d-one'] = tsne_results[:, 0]
    df_subset['tsne-2d-two'] = tsne_results[:, 1]


def plot(df_subset, dim_pairs, num_labels=100, hue="y", title="", file_name="", legend=False):
    fig, ax = plt.subplots(1, 2, figsize=(16, 7), dpi=200)
    for i, (x, y) in enumerate(dim_pairs):
        sns.scatterplot(
            x=x, y=y,
            hue=hue,
            palette=sns.color_palette("hls", num_labels),
            data=df_subset,
            legend=legend,
            s=5,
            ec=None,
            ax=ax[i]
        )
    sns.despine(offset=10, trim=True)
    if title:
        fig.suptitle(title, fontsize=16)
    if file_name:
        fig.savefig(file_name, facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor(), bbox_inches='tight')


def compare_representations(experiment, table, dataset="val"):
    data_loaders, model, trainer = (table & experiment.get_restrictions()).load_model(include_trainer=True,
                                                                                      include_state_dict=True,
                                                                                      seed=42)
    num_vis_samples = 1000000
    sample_loader = torch.utils.data.DataLoader(
        data_loaders[dataset].dataset, sampler=data_loaders[dataset].sampler, batch_size=64, shuffle=False,
        num_workers=1, pin_memory=False,
    )
    print('==> Computing Representations', flush=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(42)
    np.random.seed(42)
    # Model
    print('==> Building model..', flush=True)
    model = model.to(device)
    if device == 'cuda':
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(42)

    criterion = nn.CrossEntropyLoss()
    # Representations form clean data:
    print("Clean input", flush=True)
    outputs = compute_representation(model, criterion, device, sample_loader, [])
    # Representations from noisy data:
    print("Noisy input", flush=True)
    experiment.trainer.noise_std = {0.5: 1.0}
    main_loop_modules = [
        NoiseAugmentation(config=experiment.trainer, device=device, data_loader=sample_loader, seed=42)]
    noise_outputs = compute_representation(model, criterion, device, sample_loader, main_loop_modules)

    # Compare
    cosine_criterion = nn.CosineEmbeddingLoss()
    cosine = cosine_criterion(outputs, noise_outputs, torch.ones(outputs.shape[:1], device=device))
    mse_criterion = nn.MSELoss()
    mse = mse_criterion(outputs, noise_outputs)
    print("Clean vs. Noise: Cosine loss:", cosine.item(), "MSE loss:", mse.item(), flush=True)
    outputs = outputs.cpu()
    noise_outputs = noise_outputs.cpu()

    torch.manual_seed(42)
    np.random.seed(42)
    if device == 'cuda':
        torch.cuda.manual_seed(42)
    targets = torch.cat([t for _, t in sample_loader]).cpu()

    combined_df = pd.DataFrame()
    for noise_level, rep in (
            (0.0, outputs),
            (0.1, noise_outputs),
    ):
        feat_cols = ['dim' + str(i) for i in range(rep.shape[1])]
        df = pd.DataFrame(rep, columns=feat_cols)
        df['y'] = targets
        df['label'] = df['y'].apply(lambda i: str(i))
        df['noise'] = np.ones_like(targets) * noise_level
        combined_df = combined_df.append(df, ignore_index=True)

        # For reproducability of the results
        np.random.seed(42)
        rndperm = np.random.permutation(df.shape[0])

        df_subset = df.loc[rndperm[:num_vis_samples], :].copy()
        data_subset = df_subset[feat_cols].values

        compute_pca(df_subset, data_subset)
        compute_tsne(df_subset, data_subset)
        plot(df_subset, (
            ("pca-one", "pca-two"),
            ("tsne-2d-one", "tsne-2d-two"),),
             num_labels=min(num_vis_samples, max(targets) + 1),
             title=(
                       "Representations from noisy data " if noise_level > 0 else "Representations from clean data ") + experiment.description,
             file_name=("noise" if noise_level > 0 else "clean") + make_hash(experiment.get_key())
             )

    # For reproducability of the results
    np.random.seed(42)
    rndperm = np.random.permutation(df.shape[0])

    df_subset = combined_df.loc[rndperm[:num_vis_samples], :].copy()
    df_subset = df_subset.append(combined_df.loc[df.shape[0] + rndperm[:num_vis_samples], :].copy())
    data_subset = df_subset[feat_cols].values

    compute_pca(df_subset, data_subset)
    compute_tsne(df_subset, data_subset)
    plot(df_subset, (
        ("pca-one", "pca-two"),
        ("tsne-2d-one", "tsne-2d-two"),),
         num_labels=2,
         hue="noise",
         title="Noise vs Clean " + experiment.description,
         file_name="noise_vs_clean" + make_hash(experiment.get_key()),
         legend="brief")

    return df_subset