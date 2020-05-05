from itertools import cycle
from bias_transfer.utils.io import save_checkpoint
from mlutils.training import copy_state


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
    best_state_dict = copy_state(model)

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

    finalize(model, best_state_dict)


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
