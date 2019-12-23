import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
from bias_transfer.trainer import apply_noise, load_model
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib import cm
from sklearn.cluster import AgglomerativeClustering


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
                    inputs_ = apply_noise(inputs.clone(), device, std=noise_std, snr=None)
                else:
                    inputs_ = inputs.clone()
                inputs_ = inputs_.transpose(1, 3)
                axs[batch_idx, i].imshow(inputs_[0, :].cpu().numpy())
                if batch_idx == 0:
                    axs[batch_idx, i].set_title('{}'.format(noise_std))
            for i, noise_snr in enumerate(noise_snrs):
                if add_noise and noise_snr:
                    inputs_ = apply_noise(inputs.clone(), device, std=None, snr=noise_snr)
                else:
                    inputs_ = inputs.clone()
                inputs_ = inputs_.transpose(1, 3)
                axs[batch_idx, i + len(noise_stds)].imshow(inputs_[0, :].cpu().numpy())
                if batch_idx == 0:
                    axs[batch_idx, i + len(noise_stds)].set_title('{}'.format(noise_snr))
    return fig

def visualize_corr_matrix(model, data_loader,
                          num_samples: int = 2,
                          layer: int =0,
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
        model, _, _ = load_model(model, load_from_path)
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
            clusters = [AgglomerativeClustering(n_clusters=12).fit(1-mat[i].detach().cpu()) for i
                        in range(0, len(mat))]
            indices = [np.argsort(cluster.labels_) for cluster in clusters]
            axs[batch_idx].matshow(mat[layer][indices[layer]][:, indices[layer]].detach().cpu(), cmap=cm.get_cmap(name="Spectral_r"))
            axs[batch_idx].matshow(mat[layer].detach().cpu(), cmap=cm.get_cmap(name="Spectral_r"))
            # f.suptitle("Layer " + str(i) + " Batch-wise Correlation Similarity Matrices", fontsize=20)
    return fig
