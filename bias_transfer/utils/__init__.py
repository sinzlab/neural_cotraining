from torch import nn
from itertools import cycle


def get_subdict(dictionary, keys=None):
    if keys:
        return {k: v for k, v in dictionary.items() if k in keys}
    return dictionary


class StopClosureWrapper:
    def __init__(self, stop_closures):
        self.stop_closures = stop_closures

    def __call__(self, model):
        results = {}
        for k in self.stop_closures:
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

    def _objective():
        if switch_mode:
            model.eval()
        ret = objective_closure(model)
        if switch_mode:
            model.train(training_status)
        return ret

    def finalize(model, best_state_dict):
        old_objective = _objective()
        if restore_best:
            model.load_state_dict(best_state_dict)
            print(
                "Restoring best model! {} ---> {}".format(old_objective, _objective())
            )
        else:
            print("Final best model! objective {}".format(_objective()))

    best_objective = current_objective = _objective()
    for epoch in range(start, max_iter):

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

    finalize(model, best_state_dict)


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


def stringify(x):
    if type(x) is dict:
        x = ".".join(["{}_{}".format(k, v) for k, v in x.items()])
    return str(x)


def weight_reset(m):
    if (
        isinstance(m, nn.Conv2d)
        or isinstance(m, nn.Linear)
        or isinstance(m, nn.Conv3d)
        or isinstance(m, nn.ConvTranspose1d)
        or isinstance(m, nn.ConvTranspose2d)
        or isinstance(m, nn.ConvTranspose3d)
        or isinstance(m, nn.BatchNorm1d)
        or isinstance(m, nn.BatchNorm2d)
        or isinstance(m, nn.BatchNorm3d)
    ):
        m.reset_parameters()
