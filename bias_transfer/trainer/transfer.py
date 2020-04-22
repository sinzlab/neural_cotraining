import torch
from torch import nn
from torch.utils.data import TensorDataset

from bias_transfer.dataset.combined_dataset import CombinedDataset, JoinedDataset
from bias_transfer.utils import weight_reset
from bias_transfer.utils.io import load_model
from bias_transfer.trainer.main_loop import main_loop



def compute_representation(model, criterion, device, data_loader, rep_name):
    acc, loss, module_losses, collected_outputs = main_loop(
        model=model,
        criterion=criterion,
        device=device,
        optimizer=None,
        data_loader=data_loader,
        epoch=0,
        n_iterations=len(data_loader),
        modules=[],
        train_mode=False,
        return_outputs=True,
        neural_prediction=False,  # TODO we will use neural prediction in the future
    )
    outputs = [o[rep_name] for o in collected_outputs]
    return torch.cat(outputs)


def generate_rep_dataset(model, criterion, device, data_loader, rep_name):
    data_loader_ = torch.utils.data.DataLoader(
        data_loader.dataset,
        batch_size=data_loader.batch_size,
        sampler=None,  # make sure the dataset is in the right order and complete
        num_workers=data_loader.num_workers,
        pin_memory=data_loader.pin_memory,
        shuffle=False,
    )
    representation = compute_representation(
        model, criterion, device, data_loader_, rep_name
    )
    rep_dataset = TensorDataset(representation.to("cpu"))
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
        shuffle=False,
    )
    return combined_data_loader


def transfer_model(to_model, config, criterion=None, device=None, data_loader=None):
    model = load_model(config.transfer_from_path, to_model, ignore_missing=True)
    if config.rdm_transfer:
        data_loader = generate_rep_dataset(
            model, criterion, device, data_loader, "conv_rep"
        )
        model.apply(
            weight_reset
        )  # model was only used to generated representations now we clear it again
    else:
        if config.reset_linear:
            model.readout.apply(weight_reset)
        if config.freeze:
            model.freeze(config.freeze)
    return data_loader
