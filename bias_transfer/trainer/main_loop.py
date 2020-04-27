import torch
from tqdm import tqdm
import numpy as np

from bias_transfer.utils import LongCycler
from nnvision.utility.measures import get_correlations


def neural_full_objective(
    model, outputs, dataloader, criterion, scale_loss, data_key, inputs, targets
):
    loss_scale = (
        np.sqrt(len(dataloader[data_key].dataset) / inputs.shape[0])
        if scale_loss
        else 1.0
    )
    loss = criterion(outputs, targets)
    return loss_scale * loss + model.regularizer(data_key)


def move_data(batch_data, device):
    batch_dict = None
    data_key, inputs= batch_data[0], batch_data[1][0]

    if len(batch_data[1]) > 2:
        targets = [b.to(device) for b in batch_data[1][1:]]
    else:
        targets = batch_data[1][1].to(device)

    if data_key != 'img_classification':
        inputs, targets = (
            inputs.to(device),
            targets.to(device),
        )
        batch_dict = {data_key: [(inputs, targets)]}
        return inputs, targets, data_key, batch_dict
    inputs = inputs.to(device, dtype=torch.float)
    return inputs, targets, data_key, batch_dict



def main_loop(
    model,
    criterion,
    device,
    optimizer,
    data_loader,
    n_iterations,
    modules,
    epoch: int = 0,
    train_mode=True,
    return_outputs=False,
    scale_loss=True,
    optim_step_count=1,
    eval_type="Validation",
    return_eval=False,
):

    model.train() if train_mode else model.eval()
    task_dict, correct, total, module_losses, collected_outputs = {}, 0, {}, {}, []
    for k in criterion:
        task_dict[k] = {'epoch_loss': 0, 'eval': 0}
        total[k] = 0
    for module in modules:
        if module.criterion:  # some modules may compute an additonal output/loss
            module_losses[module.__class__.__name__] = 0
    if hasattr(
        tqdm, "_instances"
    ):  # To have tqdm output without line-breaks between steps
        tqdm._instances.clear()
    with torch.enable_grad() if train_mode else torch.no_grad():

        with tqdm(
            enumerate(LongCycler(data_loader)),
            total=n_iterations,
            desc="{}".format("Train" if train_mode else eval_type)
            if return_eval
            else "{} Epoch {}".format("Train" if train_mode else eval_type, epoch),
        ) as t:

            for module in modules:
                module.pre_epoch(model, train_mode, epoch)

            if train_mode:
                optimizer.zero_grad()

            for batch_idx, batch_data in t:
                # Pre-Forward
                loss = torch.zeros(1, device=device)
                inputs, targets, data_key, batch_dict = move_data(
                    batch_data, device
                )
                shared_memory = {}  # e.g. to remember where which noise was applied
                model_ = model
                for module in modules:
                    model_, inputs = module.pre_forward(
                        model_, inputs, shared_memory, train_mode=train_mode, data_key=data_key
                    )
                # Forward
                outputs = model_(inputs)
                # Post-Forward and Book-keeping
                def average_loss(loss_):
                    return loss_ / (batch_idx + 1)

                for module in modules:
                    outputs, loss, targets = module.post_forward(
                        outputs,
                        loss,
                        targets,
                        module_losses,
                        train_mode,
                        **shared_memory
                    )
                if return_outputs:
                    collected_outputs.append(outputs)
                if data_key != 'img_classification':
                    loss += neural_full_objective(
                        model,
                        outputs,
                        data_loader,
                        criterion['neural'],
                        scale_loss,
                        data_key,
                        inputs,
                        targets,
                    )
                    total['neural'] += get_correlations(
                        model,
                        batch_dict,
                        device=device,
                        as_dict=False,
                        per_neuron=False,
                    )
                    task_dict['neural']['eval'] = average_loss(total['neural'])
                else:
                    loss += criterion['img_classification'](outputs["logits"], targets)
                    _, predicted = outputs["logits"].max(1)
                    total['img_classification'] += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    task_dict['img_classification']['eval'] = 100.0 * correct / total['img_classification']

                task_dict[data_key]['epoch_loss'] = average_loss(task_dict[data_key]['epoch_loss'] + loss.item())


                t.set_postfix(
                    **task_dict,
                    **{k: average_loss(l) for k, l in module_losses.items()}
                )
                if train_mode:
                    # Backward
                    loss.backward()
                    for module in modules:
                        module.post_backward(model)
                    if (batch_idx + 1) % optim_step_count == 0:
                        optimizer.step()
                        optimizer.zero_grad()

    if return_outputs:
        return (
            task_dict,
            {k: average_loss(l) for k, l in module_losses.items()},
            collected_outputs,
        )

    if return_eval:
        return list(task_dict.values())[0]['eval']

    return (
        task_dict,
        {k: average_loss(l) for k, l in module_losses.items()},
    )
