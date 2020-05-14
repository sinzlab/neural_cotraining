from itertools import cycle
from bias_transfer.utils.io import save_checkpoint
from mlutils.training import copy_state
import torch
from torch import nn


def get_subdict(dictionary, keys=None):
    if keys:
        return {k: v for k, v in dictionary.items() if k in keys}
    return dictionary


class StopClosureWrapper:
    def __init__(self, stop_closures):
        self.stop_closures = stop_closures

    def __call__(self, model):
        results = {}
        for k in self.stop_closures.keys():
            results[k] = self.stop_closures[k](model)
        return results


def fixed_training_process(
    model,
    stop_closures,
    config,
    start=0,
    max_iter=1000,
    switch_mode=True,
    restore_best=True,
    scheduler=None,
):

    training_status = model.training
    objective_closure = StopClosureWrapper(stop_closures)
    best_state_dict = copy_state(model)

    def _objective():
        if switch_mode:
            model.eval()
        ret = objective_closure(model)
        if switch_mode:
            model.train(training_status)
        return ret

    def finalize(model, best_state_dict, old_objective, best_objective):
        if restore_best:
            model.load_state_dict(best_state_dict)
            print(
                "Restoring best model! {} ---> {}".format(
                    list(old_objective.values())[0], list(best_objective.values())[0]
                )
            )
        else:
            print(
                "Final best model! objective {}".format(
                    list(best_objective.values())[0]
                )
            )

    best_objective = current_objective = _objective()
    for epoch in range(start + 1, max_iter + 1):

        yield epoch, current_objective

        current_objective = _objective()

        # if a scheduler is defined, a .step with the current objective is all that is needed to reduce the LR
        if scheduler is not None:
            if config.adaptive_lr:
                scheduler.step(list(current_objective.values())[0])
            else:
                scheduler.step(epoch=epoch)
        print(
            "Validation Epoch {} -------> {}".format(epoch, current_objective),
            flush=True,
        )

        is_better = (
            True if current_objective[k] > best_objective[k] else False
            for k in current_objective.keys()
        )
        if all(is_better):
            best_state_dict = copy_state(model)
            best_objective = current_objective

    finalize(model, best_state_dict, current_objective, best_objective)


def save_best_model(model, optimizer, dev_eval, epoch, best_eval, best_epoch, uid):
    if isinstance(dev_eval, dict):
        is_better = (
            True if dev_eval[k] > best_eval[k] else False for k in dev_eval.keys()
        )
    else:
        is_better = [dev_eval > list(best_eval.values())[0]]
    if all(is_better):
        save_checkpoint(
            model,
            optimizer,
            dev_eval,
            epoch - 1,
            "./checkpoint",
            "ckpt.{}.pth".format(uid),
        )
        best_eval = (
            dev_eval if isinstance(dev_eval, dict) else {k: dev_eval for k in best_eval}
        )
        best_epoch = epoch - 1
    return best_epoch, best_eval


class MTL_Cycler:
    def __init__(self, loaders, main_key="img_classification", ratio=1):
        self.main_key = main_key
        self.main_loader = loaders[main_key]
        self.other_loaders = {k: loaders[k] for k in loaders.keys() if k != main_key}
        self.ratio = ratio
        self.num_batches = len(self.main_loader) * (ratio + 1)

    def generate_batch(self, main_cycle, other_cycles_dict):
        for i in range(self.num_batches):
            yield self.main_key, main_cycle
            for _ in range(self.ratio):
                key, loader = next(other_cycles_dict)
                yield key, loader

    def __iter__(self):
        other_cycles = {k: cycle(v) for k, v in self.other_loaders.items()}
        other_cycles_dict = cycle(other_cycles.items())
        main_cycle = cycle(self.main_loader)
        for k, loader in self.generate_batch(main_cycle, other_cycles_dict):
            yield k, next(loader)

    def __len__(self):
        return self.num_batches


class LongCycler:
    """
    Cycles through trainloaders until the loader with largest size is exhausted.
        Needed for dataloaders of unequal size (as in the monkey data).
    """

    def __init__(self, loaders):
        self.loaders = loaders
        self.max_batches = max([len(loader) for loader in self.loaders.values()])

    def __iter__(self):
        cycles = [cycle(loader) for loader in self.loaders.values()]
        for k, loader, _ in zip(
            cycle(self.loaders.keys()),
            (cycle(cycles)),
            range(len(self.loaders) * self.max_batches),
        ):
            yield k, next(loader)

    def __len__(self):
        return len(self.loaders) * self.max_batches


class XEntropyLossWrapper(nn.Module):
    def __init__(self, criterion):
        super(XEntropyLossWrapper, self).__init__()
        self.log_var = nn.Parameter(torch.zeros(1))
        self.criterion = criterion  # it is nn.CrossEntropyLoss

    def forward(self, preds, targets):
        precision = torch.exp(-self.log_var)
        loss = precision * self.criterion(preds, targets) + self.log_var
        return loss  # , self.log_var.item()


class NBLossWrapper(nn.Module):
    def __init__(self):
        super(NBLossWrapper, self).__init__()
        self.log_r = nn.Parameter(torch.zeros(1))

    def forward(self, preds, targets):
        r = torch.exp(self.log_r)
        loss = (
            (targets + r) * torch.log(preds + r)
            - (targets * torch.log(preds))
            - (r * self.log_r)
            + torch.lgamma(r)
            - torch.lgamma(targets + r)
            + torch.lgamma(targets + 1)
            + 1e-5
        )
        return loss.mean()  # , self.log_r.item()
