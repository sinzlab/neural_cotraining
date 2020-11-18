from itertools import cycle
from bias_transfer.utils.io import save_checkpoint
from mlutils.training import copy_state
import torch
from torch import nn
import numpy as np

def get_subdict(dictionary:dict, keys:list=None):
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

    def __call__(self, model):
        results = {task: {} for task in self.stop_closures.keys() }
        for task in self.stop_closures.keys():
            if task != "img_classification":
                for objective in self.stop_closures[task].keys():
                    results[task][objective] = self.stop_closures[task][objective](model)
            else:
                res, _ = self.stop_closures[task](model)
                results[task]['eval'] = res[task]['eval']
                results[task]['loss'] = res[task]['epoch_loss']
        return results

def map_to_task_dict(task_dict, fn):
    """
    Args:
        task_dict: dictionary of the form: e.g. {"img_classification": {"loss": 0.2} }
        fn: function to apply to all values in task_dict
    Return:
        numpy_array: array of booleans as result of applying fn to all values
    """
    result = [ fn(task_dict[task][objective]) for task in task_dict.keys()
               for objective in task_dict[task].keys()]
    return np.array(result)



def early_stopping(
        model,
        uid,
        objective_closure,
        config,
        interval=5,
        patience=20,
        start=0,
        max_iter=1000,
        maximize=True,
        tolerance=1e-5,
        switch_mode=True,
        restore_best=True,
        tracker=None,
        scheduler=None,
        lr_decay_steps=1,
):

    training_status = model.training
    objective_closure = StopClosureWrapper(objective_closure)

    def _objective():
        if switch_mode:
            model.eval()
        ret = objective_closure(model)
        if switch_mode:
            model.train(training_status)
        return ret

    def decay_lr(model, best_state_dict, old_objective, best_objective):
        if restore_best:
            model.load_state_dict(best_state_dict)
            print("Restoring best model after lr decay! {} ---> {}".format(old_objective, best_objective), flush=True)

    def finalize(model, best_state_dict, old_objective, best_objective):
        if restore_best:
            model.load_state_dict(best_state_dict)
            print(
                "Restoring best model! {} ---> {}".format(
                    old_objective, best_objective
                )
            )
        else:
            print(
                "Final best model! objective {}".format(
                    best_objective
                )
            )


    epoch = start
    # turn into a sign
    maximize = -1 if maximize else 1
    best_objective = current_objective = _objective()
    best_state_dict = copy_state(model)

    if scheduler is not None:
        if (config.scheduler == "adaptive") and (not config.scheduler_options['mtl']):  # only works sofar with one task but not with MTL
            scheduler.step(current_objective[config.to_monitor[0]]['eval' if config.maximize else 'loss'])

    for repeat in range(lr_decay_steps):
        patience_counter = -1

        while patience_counter < patience and epoch < max_iter:

            for _ in range(interval):
                epoch += 1
                if tracker is not None:
                    tracker.log_objective(current_objective)

                def isnotfinite(score):
                    return ~np.isfinite(score)

                if (map_to_task_dict(current_objective, isnotfinite)).any():
                    print("Objective is not Finite. Stopping training")
                    finalize(model, best_state_dict, current_objective, best_objective)
                    return
                yield epoch, current_objective

            current_objective = _objective()

            # if a scheduler is defined, a .step with the current objective is all that is needed to reduce the LR
            if scheduler is not None:
                if (config.scheduler == "adaptive") and (not config.scheduler_options['mtl']):   # only works sofar with one task but not with MTL
                    scheduler.step(current_objective[config.to_monitor[0]]['eval' if config.maximize else 'loss'])
                elif config.scheduler == "manual":
                    scheduler.step()

            def test_current_obj(obj, best_obj):
                obj_key = 'eval' if config.maximize else 'loss'
                result = [ obj[task][obj_key] * maximize < best_obj[task][obj_key] * maximize - tolerance for task in obj.keys()
                           if task in config.to_monitor ]
                return np.array(result)

            if (test_current_obj(current_objective, best_objective)).all():
                print(
                    "Validation [{:03d}|{:02d}/{:02d}] ---> {}".format(epoch, patience_counter, patience, current_objective),
                    flush=True,
                )
                best_state_dict = copy_state(model)
                best_objective = current_objective
                patience_counter = -1
            else:
                patience_counter += 1
                print(
                    "Validation [{:03d}|{:02d}/{:02d}] -/-> {}".format(epoch, patience_counter, patience, current_objective),
                    flush=True,
                )

            if (config.scheduler == "manual") and (epoch in config.scheduler_options['milestones']):
                decay_lr(model, best_state_dict, current_objective, best_objective)


        if (epoch < max_iter) & (lr_decay_steps > 1) & (repeat < lr_decay_steps):
            if (config.scheduler == "adaptive") and (config.scheduler_options['mtl']):   #adaptive lr scheduling for mtl alongside early_stopping
                scheduler.step()
            decay_lr(model, best_state_dict, current_objective, best_objective)

    finalize(model, best_state_dict, current_objective, best_objective)


class MTL_Cycler:
    def __init__(self, loaders, main_key="img_classification", ratio=1):
        self.main_key = main_key  # data_key of the dataset whose batch ratio is always 1
        self.main_loader = loaders[main_key]
        self.other_loaders = {k: loaders[k] for k in loaders.keys() if k != main_key}
        self.ratio = ratio   # number of neural batches vs. one batch from TIN
        self.num_batches = int(len(self.main_loader) * (ratio + 1))
        self.backward = False

    def generate_batch(self, main_cycle, other_cycles_dict):
        if self.ratio >= 1 or self.ratio == 0:
            for main_batch_idx in range(len(self.main_loader)):
                for batch_idx in range(self.ratio):
                    key, loader = next(other_cycles_dict)
                    yield key, loader
                self.backward = True
                yield self.main_key, main_cycle
                self.backward = False
        else:
            for i in range(len(self.main_loader)):
                if (i+1) % (1/self.ratio) == 0:
                    key, loader = next(other_cycles_dict)
                    yield key, loader
                self.backward = True
                yield self.main_key, main_cycle
                self.backward = False

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
        self.backward = False

    def __iter__(self):
        cycles = [cycle(loader) for loader in self.loaders.values()]
        grad_accum_step = len(self.loaders)
        for k, loader, batch_idx in zip(
            cycle(self.loaders.keys()),
            (cycle(cycles)),
            range(len(self.loaders) * self.max_batches),
        ):
            if (batch_idx + 1) % grad_accum_step == 0:
                self.backward = True
            yield k, next(loader)
            self.backward = False

    def __len__(self):
        return len(self.loaders) * self.max_batches


class XEntropyLossWrapper(nn.Module):
    def __init__(self, criterion):
        super(XEntropyLossWrapper, self).__init__()
        self.log_w = nn.Parameter(torch.zeros(1))  #std
        self.criterion = criterion  # it is nn.CrossEntropyLoss

    def forward(self, preds, targets):
        precision = torch.exp(-self.log_w)
        loss = precision * self.criterion(preds, targets) + self.log_w
        return loss


class NBLossWrapper(nn.Module):
    def __init__(self, loss_sum):
        super(NBLossWrapper, self).__init__()
        self.log_w = nn.Parameter(torch.zeros(1)) #r: number of successes
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

def set_bn_to_eval(model, freeze_bn):
    if not freeze_bn['mtl']:
        for name, param in model.features.named_children():
            if int(name) < freeze_bn['last_layer'] and "BatchNorm" in param.__class__.__name__:
                param.train(False)
    else:
        for name, param in model.mtl_vgg_core.shared_block.named_children():
            if int(name) < freeze_bn['last_layer'] and "BatchNorm" in param.__class__.__name__:
                param.train(False)