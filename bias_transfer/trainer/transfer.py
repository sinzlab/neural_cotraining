import torch
from torch import nn
from torch.utils.data import TensorDataset

from bias_transfer.dataset.combined_dataset import CombinedDataset, JoinedDataset
from bias_transfer.utils.io import load_model
from bias_transfer.trainer.main_loop import main_loop


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


def compute_representation(model, criterion, device, data_loader, rep_name):
    acc, loss, module_losses, collected_outputs = main_loop(
        model=model,
        criterion=criterion,
        device=device,
        optimizer=None,
        data_loader=data_loader,
        epoch=0,
        modules=[],
        train_mode=False,
        return_outputs=True,
    )
    outputs = [o[rep_name] for o in collected_outputs]
    return torch.cat(outputs)


def generate_rep_dataset(model, criterion, device, data_loader, rep_name):
    representation = compute_representation(
        model, criterion, device, data_loader, rep_name
    )
    rep_dataset = TensorDataset(representation)
    img_dataset = data_loader.dataset
    combined_dataset = CombinedDataset(
        JoinedDataset(
            sample_datasets=[img_dataset], target_datasets=[img_dataset, rep_dataset]
        )
    )
    combined_data_loader = torch.utils.data.DataLoader(
        dataset=combined_dataset,
        batch_size=data_loader.batch_size,
        sampler=data_loader.sampler,
        num_workers=data_loader.num_workers,
        pin_memory=data_loader.pin_memory,
        shuffle=data_loader.shuffle,
    )
    return combined_data_loader


def transfer_model(to_model, config, criterion=None, device=None, data_loader=None):
    model = load_model(config.transfer_from_path, to_model, ignore_missing=True)
    if config.rdm_transfer:
        data_loader = generate_rep_dataset(model, criterion, device, data_loader, "conv_rep")
        model.apply(
            weight_reset
        )  # model was only used to generated representations now we clear it again
    else:
        if config.reset_linear:
            model.readout.apply(weight_reset)
        if config.freeze:
            model.freeze(config.freeze)
    return data_loader
