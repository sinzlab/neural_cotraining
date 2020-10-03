import torch
from tqdm import tqdm
import numpy as np

from bias_transfer.trainer import utils as uts
from nnvision.utility.measures import get_correlations


def neural_full_objective(
    model, outputs, dataloader, criterion, scale_loss, data_key, inputs, targets
):

    loss = criterion(outputs, targets)
    loss_scale = (
        np.sqrt(len(dataloader[data_key].dataset) / inputs.shape[0])
        if scale_loss
        else 1.0
    )
    loss *= loss_scale
    if scale_loss:
        loss += model.regularizer(data_key)
    return loss


def move_data(batch_data, device):
    if len(batch_data[1]) == 3:
        data_key, (inputs, labels, responses) = batch_data
    else:
        data_key, (inputs, labels) = batch_data
    inputs = inputs.to(device, dtype=torch.float)
    labels = labels.to(device)
    if len(batch_data[1]) == 2:
        return inputs, labels, {}, data_key, {}
    responses = responses.to(device)
    batch_dict = {data_key: [(inputs, labels, responses)]}
    return inputs, labels, responses, data_key, batch_dict


def main_loop(
    model,
    criterion,
    device,
    optimizer,
    data_loader,
    modules,
    scale_loss,
    epoch: int = 0,
    train_mode=True,
    return_outputs=False,
    eval_type="Validation",
    cycler="LongCycler",
    cycler_args={},
    loss_weighing=False,
):
    model.train() if train_mode else model.eval()
    task_dict = {}
    correct = 0
    if loss_weighing:
        total_loss_weight = {}
    total = {}
    total_loss = {}
    module_losses = {}
    collected_outputs = []
    for k in criterion:
        task_dict[k] = {"epoch_loss": 0, "eval": 0}
        if loss_weighing:
            task_dict[k]['loss_weight'] = 0
            total_loss_weight[k] = 0
        total[k] = 0
        total_loss[k] = 0
    for module in modules:
        if module.criterion:  # some modules may compute an additonal output/loss
            module_losses[module.__class__.__name__] = 0

    if cycler_args:
        cycler_args = dict(cycler_args)
        cycler_args['ratio'] = cycler_args['ratio'][max(i for i in list(cycler_args['ratio'].keys()) if epoch >= i)]

    data_cycler = getattr(uts, cycler)(data_loader, **cycler_args)
    n_iterations = len(data_cycler)

    if hasattr(
        tqdm, "_instances"
    ):  # To have tqdm output without line-breaks between steps
        tqdm._instances.clear()
    with torch.enable_grad() if train_mode else torch.no_grad():

        with tqdm(
            enumerate(data_cycler),
            total=n_iterations,
            desc="{} Epoch {}".format("Train" if train_mode else eval_type, epoch),
        ) as t:

            for module in modules:
                module.pre_epoch(model, train_mode, epoch, optimizer=optimizer)

            if train_mode:
                optimizer.zero_grad()
            loss = {}

            for batch_idx, batch_data in t:
                # Pre-Forward
                loss = torch.zeros(1, device=device)
                neural_loss = torch.zeros(1, device=device)
                inputs, labels, responses, data_key, batch_dict = move_data(batch_data, device)
                shared_memory = {}  # e.g. to remember where which noise was applied
                model_ = model
                for module in modules:
                    model_, inputs = module.pre_forward(
                        model_,
                        inputs,
                        shared_memory,
                        train_mode=train_mode,
                        data_key=data_key,
                    )
                # Forward
                labels_pred, responses_pred = model_(inputs)
                # Post-Forward and Book-keeping
                def average_loss(loss_):
                    return loss_ / (batch_idx + 1)

                #if return_outputs:
                #    collected_outputs.append(outputs[0])
                for module in modules:
                    labels_pred, loss, labels = module.post_forward(
                        labels_pred,
                        loss,
                        labels,
                        module_losses,
                        train_mode,
                        **shared_memory
                    )
                for j, k in enumerate(criterion.keys()):
                    if k != "img_classification":
                        neural_loss += neural_full_objective(
                            model,
                            responses_pred,
                            data_loader,
                            criterion["neural"],
                            scale_loss,
                            data_key,
                            inputs,
                            responses,
                        )
                        total["neural"] += get_correlations(
                            model,
                            batch_dict,
                            device=device,
                            as_dict=False,
                            per_neuron=False,
                        )
                        task_dict["neural"]["eval"] = average_loss(total["neural"])
                        total_loss["neural"] += neural_loss.item()
                        task_dict["neural"]["epoch_loss"] = average_loss(
                            total_loss["neural"]
                        )
                        if loss_weighing:
                            total_loss_weight["neural"] += np.exp(criterion["neural"].log_w.item())
                            task_dict["neural"]["loss_weight"] = average_loss(
                                total_loss_weight["neural"]
                            )

                        if train_mode:
                            # Backward
                            neural_loss.backward()
                            for module in modules:
                                module.post_backward(model)

                    else:
                        loss += criterion["img_classification"](labels_pred, labels)
                        _, predicted = labels_pred.max(1)
                        total["img_classification"] += labels.size(0)
                        correct += predicted.eq(labels).sum().item()
                        task_dict["img_classification"]["eval"] = (
                            100.0 * correct / total["img_classification"]
                        )
                        total_loss["img_classification"] += loss.item()
                        task_dict["img_classification"]["epoch_loss"] = average_loss(
                            total_loss["img_classification"]
                        )
                        if loss_weighing:
                            total_loss_weight["img_classification"] += np.exp(criterion["img_classification"].log_w.item())
                            task_dict["img_classification"]["loss_weight"] = average_loss(
                                total_loss_weight["img_classification"]
                            )
                        if train_mode:
                            # Backward
                            loss.backward(retain_graph=True)
                            for module in modules:
                                module.post_backward(model)

                if train_mode:
                    #if data_cycler.backward:
                    optimizer.step()
                    optimizer.zero_grad()

                t.set_postfix(
                    **{
                        task: {obj: round(value, 3) for obj, value in res.items() if obj != "loss_weight"}
                        for task, res in task_dict.items()
                    },
                    **{k: average_loss(l) for k, l in module_losses.items()}
                )


    # if return_outputs:
    #     return (
    #         task_dict,
    #         {k: average_loss(l) for k, l in module_losses.items()},
    #         collected_outputs,
    #     )

    return (
        task_dict,
        {k: average_loss(l) for k, l in module_losses.items()},
    )
