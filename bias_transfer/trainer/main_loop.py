import torch
from tqdm import tqdm
import numpy as np

from bias_transfer.trainer import utils as uts
from functools import partial
from .utils import set_bn_to_eval

def neural_full_objective(
    model, outputs, dataloader, criterion, scale_loss, data_key, inputs, targets, multi, neural_set, mtl
):

    loss = criterion(outputs, targets)
    loss_scale = (
        np.sqrt(len(dataloader[neural_set][data_key].dataset) / inputs.shape[0])
        if scale_loss
        else 1.0
    )
    loss *= loss_scale
    if scale_loss:
        if not multi:
            if mtl:
                loss += model.regularizer(neural_set, data_key)
            else:
                loss += model.regularizer(data_key)
        else:
            if mtl:
                loss += model.module.regularizer(neural_set, data_key)
            else:
                loss += model.module.regularizer(data_key)
    return loss


def move_data(batch_data, tasks, device):
    data_key, inputs = batch_data[0], batch_data[1][0]
    if isinstance(data_key, tuple):
        neural_set = data_key[0]
        data_key = data_key[1]
    else:
        neural_set = None
    targets = dict()
    if len(batch_data[1]) > 2:
        for task, output_name in tasks.items():
            targets[task] = getattr(batch_data[1], output_name).to(device)
        inputs = inputs.to(device)
    else:
        if data_key == "img_classification" or ('v1' not in list(tasks.keys()) and 'v4' not in list(tasks.keys())) :
            targets['img_classification'] = batch_data[1][1].to(device)
            if isinstance(inputs, list):
                inputs = torch.cat(inputs)
            inputs = inputs.to(device, dtype=torch.float)
        else:
            targets[neural_set] = batch_data[1][1].to(device)
            inputs = inputs.to(device)
    return inputs, targets, data_key, neural_set


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
    loss_weighing=False, multi=False, mtl=False,
    freeze_bn={'last_layer': -1}
):
    model.train() if train_mode else model.eval()
    if train_mode and freeze_bn['last_layer'] > 0:
        set_bn_to_eval(model, freeze_bn, multi, [task for task in list(criterion.keys()) if task in ['v1', 'v4']])
    task_dict = {}
    correct = 0
    if loss_weighing:
        total_loss_weight = {}
    total = {}
    total_loss = {}
    module_losses = {}
    collected_outputs = []
    tasks = {"img_classification": "labels", "v1": "responses", "v4": "responses"}
    tasks = uts.get_subdict(tasks, list(criterion.keys()))
    for k in criterion:
        task_dict[k] = {"epoch_loss": 0}
        if k== 'img_classification':
            task_dict[k]['eval'] = 0
            total[k] = 0
        if loss_weighing:
            task_dict[k]['loss_weight'] = 0
            total_loss_weight[k] = 0
        total_loss[k] = 0
    for module in modules:
        if module.criterion:  # some modules may compute an additonal output/loss
            module_losses[module.__class__.__name__] = 0

    if cycler_args and cycler == "MTL_Cycler":
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

            for batch_idx, batch_data in t:
                # Pre-Forward
                loss = torch.zeros(1, device=device)
                inputs, targets, data_key, neural_set = move_data(batch_data, tasks, device)
                shared_memory = {}  # e.g. to remember where which noise was applied
                model_ = model
                for module in modules:
                    model_, inputs = module.pre_forward(
                        model_,
                        inputs,
                        shared_memory,
                        train_mode=train_mode,
                        data_key=data_key,neural_set=neural_set,
                        task_keys=list(targets.keys())
                    )
                # Forward
                outputs = model_(inputs)
                # Post-Forward and Book-keeping
                def average_loss(loss_):
                    return loss_ / (batch_idx + 1)

                if return_outputs:
                    collected_outputs.append(outputs[0])
                for module in modules:
                    outputs, loss, targets = module.post_forward(
                        outputs,
                        loss,
                        targets,
                        module_losses,
                        train_mode,
                        task_keys=list(targets.keys()),
                        **shared_memory
                    )
                if "v1" in targets.keys() or "v4" in targets.keys():
                    neural_set = "v1" if "v1" in targets.keys() else "v4"
                    loss += neural_full_objective(
                        model,
                        outputs[neural_set],
                        data_loader,
                        criterion[neural_set],
                        scale_loss,
                        data_key,
                        inputs,
                        targets[neural_set], multi=multi, neural_set=neural_set, mtl=mtl
                    )
                    total_loss[neural_set] += loss.item()
                    task_dict[neural_set]["epoch_loss"] = average_loss(
                        total_loss[neural_set]
                    )
                    if loss_weighing:
                        total_loss_weight[neural_set] += np.exp(criterion[neural_set].log_w.item())
                        task_dict[neural_set]["loss_weight"] = average_loss(
                            total_loss_weight[neural_set]
                        )

                if "img_classification" in targets.keys():
                    loss += criterion["img_classification"](outputs['img_classification'], targets['img_classification'])
                    _, predicted = outputs['img_classification'].max(1)
                    total["img_classification"] += targets['img_classification'].size(0)
                    correct += predicted.eq(targets['img_classification']).sum().item()
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

                t.set_postfix(
                    **{
                        task: {obj: round(value, 3) for obj, value in res.items() if obj != "loss_weight"}
                        for task, res in task_dict.items()
                    },
                    **{k: average_loss(l) for k, l in module_losses.items()}
                )
                if train_mode:
                    # Backward
                    loss.backward()
                    for module in modules:
                        module.post_backward(model)
                    if data_cycler.backward:
                        optimizer.step()
                        optimizer.zero_grad()

    if return_outputs:
        return (
            task_dict,
            {k: average_loss(l) for k, l in module_losses.items()},
            collected_outputs,
        )

    return (
        task_dict,
        {k: average_loss(l) for k, l in module_losses.items()},
    )
