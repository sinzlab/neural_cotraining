import torch
from torch import nn
import numpy as np


def get_subdict(dictionary: dict, keys: list = None):
    """
    Args:
        dictionary: dictionary of all keys
        keys: list of strings representing the keys to be extracted from dictionary
    Return:
        dict: subdictionary containing only input keys
    """

    if keys:
        return {k: v for k, v in dictionary.items() if k in keys}
    return dictionary


class StopClosureWrapper:
    def __init__(self, stop_closures):
        self.stop_closures = stop_closures

    def __call__(self):
        results = {task: {} for task in self.stop_closures.keys()}
        for task in self.stop_closures.keys():
            res = self.stop_closures[task]()
            results[task]["eval"] = res
        return results


def map_to_task_dict(task_dict, fn):
    """
    Args:
        task_dict: dictionary of the form: e.g. {"img_classification": {"loss": 0.2} }
        fn: function to apply to all values in task_dict
    Return:
        numpy_array: array of booleans as result of applying fn to all values
    """
    result = [
        fn(task_dict[task][objective])
        for task in task_dict.keys()
        for objective in task_dict[task].keys()
    ]
    return np.array(result)


class XEntropyLossWrapper(nn.Module):
    def __init__(self, criterion):
        super(XEntropyLossWrapper, self).__init__()
        self.log_w = nn.Parameter(torch.zeros(1))  # std corrected to var
        self.criterion = criterion  # it is nn.CrossEntropyLoss

    def forward(self, preds, targets):
        precision = torch.exp(-self.log_w)
        loss = precision * self.criterion(preds, targets) + (0.5 * self.log_w)
        return loss


class MSELossWrapper(nn.Module):
    def __init__(self, criterion):
        super(MSELossWrapper, self).__init__()
        self.log_w = nn.Parameter(torch.zeros(1))  # std corrected to var
        self.criterion = criterion  # it is .nn.MSELoss

    def forward(self, preds, targets):
        precision = torch.exp(-self.log_w)
        loss = (0.5 * precision) * self.criterion(preds, targets) + (0.5 * self.log_w)
        return loss


class NBLossWrapper(nn.Module):
    def __init__(self, loss_sum):
        super(NBLossWrapper, self).__init__()
        self.log_w = nn.Parameter(torch.zeros(1))  # r: number of successes
        self.loss_sum = loss_sum

    def forward(self, preds, targets):
        r = torch.exp(self.log_w)
        loss = (
            (targets + r) * torch.log(preds + r)
            - (targets * torch.log(preds))
            - (r * self.log_w)
            + torch.lgamma(r)
            - torch.lgamma(targets + r)
            + torch.lgamma(targets + 1)
            + 1e-5
        )
        return loss.sum() if self.loss_sum else loss.mean()


def set_bn_to_eval(model, freeze_bn, multi, tasks):
    if not freeze_bn["mtl"]:
        for name, param in model.features.named_children():
            if (
                int(name) < freeze_bn["last_layer"]
                and "BatchNorm" in param.__class__.__name__
            ):
                param.train(False)
    else:
        if not multi:
            if "v1" in tasks:
                for name, param in model.mtl_vgg_core.v1_block.named_children():
                    if (
                        int(name) < freeze_bn["last_layer"]
                        and "BatchNorm" in param.__class__.__name__
                    ):
                        param.train(False)
            if "v4" in tasks:
                for name, param in model.mtl_vgg_core.v4_block.named_children():
                    if (
                        int(name) < freeze_bn["last_layer"]
                        and "BatchNorm" in param.__class__.__name__
                    ):
                        param.train(False)
        else:
            if "v1" in tasks:
                for name, param in model.module.mtl_vgg_core.v1_block.named_children():
                    if (
                        int(name) < freeze_bn["last_layer"]
                        and "BatchNorm" in param.__class__.__name__
                    ):
                        param.train(False)
            if "v4" in tasks:
                for name, param in model.module.mtl_vgg_core.v4_block.named_children():
                    if (
                        int(name) < freeze_bn["last_layer"]
                        and "BatchNorm" in param.__class__.__name__
                    ):
                        param.train(False)


def neural_full_objective(
    model,
    outputs,
    dataloader,
    criterion,
    scale_loss,
    data_key,
    inputs,
    targets,
    multi,
    neural_set,
    mtl,
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
        if data_key == "img_classification" or (
            "v1" not in list(tasks.keys()) and "v4" not in list(tasks.keys())
        ):
            targets["img_classification"] = batch_data[1][1].to(device)
            if isinstance(inputs, list):
                inputs = torch.cat(inputs)
            inputs = inputs.to(device, dtype=torch.float)
        else:
            targets[neural_set] = batch_data[1][1].to(device)
            inputs = inputs.to(device)
    return inputs, targets, data_key, neural_set
