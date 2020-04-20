import torch
from tqdm import tqdm
import numpy as np

from mlutils.training import LongCycler
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


def move_data(batch_data, device, neural_prediction=False):
    if neural_prediction:
        data_key, (inputs, targets) = batch_data[0], batch_data[1]
        inputs, targets = (
            inputs.to(device),
            targets.to(device),
        )
        batch_dict = {data_key: [(inputs, targets)]}
    else:
        data_key = None
        batch_dict = None
        inputs = batch_data[0].to(device, dtype=torch.float)
        if len(batch_data) > 2:
            targets = [b.to(device) for b in batch_data[1:]]
        else:
            targets = batch_data[1].to(device)
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
    neural_prediction=False,
    scale_loss=True,
    optim_step_count=1,
    eval_type="Validation",
    return_eval=False,
):

    model.train() if train_mode else model.eval()
    epoch_loss, correct, total, module_losses, collected_outputs = 0, 0, 0, {}, []
    for module in modules:
        if module.criterion:  # some modules may compute an additonal output/loss
            module_losses[module.__class__.__name__] = 0
    if hasattr(
        tqdm, "_instances"
    ):  # To have tqdm output without line-breaks between steps
        tqdm._instances.clear()
    with torch.enable_grad() if train_mode else torch.no_grad():

        with tqdm(
            enumerate(LongCycler(data_loader) if neural_prediction else data_loader),
            total=n_iterations,
            desc="{}".format("Train" if train_mode else eval_type)
            if return_eval
            else "{} Epoch {}".format("Train" if train_mode else eval_type, epoch),
        ) as t:

            for module in modules:
                module.pre_epoch(model, train_mode)

            if train_mode:
                optimizer.zero_grad()

            for batch_idx, batch_data in t:
                # Pre-Forward
                loss = torch.zeros(1, device=device)
                inputs, targets, data_key, batch_dict = move_data(
                    batch_data, device, neural_prediction
                )
                shared_memory = {}  # e.g. to remember where which noise was applied
                for module in modules:
                    model, inputs = module.pre_forward(
                        model, inputs, shared_memory, train_mode=train_mode
                    )
                # Forward
                if neural_prediction:
                    outputs = model(inputs, data_key)
                else:
                    outputs = model(inputs)
                # Post-Forward
                for module in modules:
                    outputs, loss, targets = module.post_forward(
                        outputs=outputs,
                        loss=loss,
                        extra_losses=module_losses,
                        train_mode=train_mode,
                        targets=targets,
                        **shared_memory
                    )
                if return_outputs:
                    collected_outputs.append(outputs)
                if neural_prediction:
                    loss += neural_full_objective(
                        model,
                        outputs,
                        data_loader,
                        criterion,
                        scale_loss,
                        data_key,
                        inputs,
                        targets,
                    )
                else:
                    loss += criterion(outputs["logits"], targets)
                epoch_loss += loss.item()
                # Book-keeping
                def average_loss(loss_):
                    return loss_ / (batch_idx + 1)

                if neural_prediction:
                    total += get_correlations(
                        model,
                        batch_dict,
                        device=device,
                        as_dict=False,
                        per_neuron=False,
                    )
                    eval = average_loss(total)
                else:
                    _, predicted = outputs["logits"].max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    eval = 100.0 * correct / total

                t.set_postfix(
                    eval=eval,
                    loss=average_loss(epoch_loss),
                    **{k: average_loss(l) for k, l in module_losses.items()}
                )
                if train_mode:
                    # Backward
                    loss.backward()
                    if (batch_idx + 1) % optim_step_count == 0:
                        optimizer.step()
                        optimizer.zero_grad()

    if return_outputs:
        return (
            eval,
            average_loss(epoch_loss),
            {k: average_loss(l) for k, l in module_losses.items()},
            collected_outputs,
        )

    if return_eval:
        return eval

    return (
        eval,
        average_loss(epoch_loss),
        {k: average_loss(l) for k, l in module_losses.items()},
    )
