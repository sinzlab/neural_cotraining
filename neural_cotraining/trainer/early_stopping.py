import numpy as np
from .utils import StopClosureWrapper, map_to_task_dict


def early_stopping(
    model,
    objective_closure,
    config,
    optimizer,
    interval=5,
    patience=20,
    max_iter=1000,
    maximize=True,
    tolerance=1e-5,
    switch_mode=True,
    restore_best=True,
    tracker=None,
    scheduler=None,
    lr_decay_steps=1,
    checkpointing=None,
):
    def _objective():
        if switch_mode:
            model.eval()
        ret = objective_closure()
        if switch_mode:
            model.train(training_status)
        return ret

    def decay_lr():
        if restore_best:
            restored_epoch, _ = checkpointing.restore(
                restore_only_state=True, action="best"
            )
            print(
                "Restoring best model from epoch {} after lr-decay!".format(
                    restored_epoch
                )
            )

    def finalize(best_objective):
        if restore_best:
            restored_epoch, _ = checkpointing.restore(
                restore_only_state=True, action="best"
            )
            print("Restoring best model from epoch! {}".format(restored_epoch))
        else:
            print("Final best model! objective {}".format(best_objective))

    training_status = model.training
    objective_closure = StopClosureWrapper(objective_closure)

    epoch, patience_counter = checkpointing.restore(action="last")
    # turn into a sign
    maximize = -1 if maximize else 1
    best_objective = current_objective = _objective()

    if scheduler is not None:
        if (config.scheduler == "adaptive") and (
            not config.scheduler_options["mtl"]
        ):  # only works sofar with one task but not with MTL
            scheduler.step(
                current_objective[config.to_monitor[0]][
                    "eval" if config.maximize else "loss"
                ]
            )

    for repeat in range(lr_decay_steps):

        while patience_counter < patience and epoch < max_iter:

            for _ in range(interval):
                epoch += 1
                if tracker is not None:
                    tracker.log_objective(current_objective)

                def isnotfinite(score):
                    return ~np.isfinite(score)

                if (map_to_task_dict(current_objective, isnotfinite)).any():
                    print("Objective is not Finite. Stopping training")
                    finalize(best_objective)
                    return
                yield epoch, current_objective

            current_objective = _objective()

            # if a scheduler is defined, a .step with the current objective is all that is needed to reduce the LR
            if scheduler is not None:
                if (config.scheduler == "adaptive") and (
                    not config.scheduler_options["mtl"]
                ):  # only works sofar with one task but not with MTL
                    scheduler.step(
                        current_objective[config.to_monitor[0]][
                            "eval" if config.maximize else "loss"
                        ]
                    )
                elif config.scheduler == "manual":
                    scheduler.step()

            def test_current_obj(obj, best_obj):
                obj_key = "eval" if config.maximize else "loss"
                result = [
                    obj[task][obj_key] * maximize
                    < best_obj[task][obj_key] * maximize - tolerance
                    for task in obj.keys()
                    if task in config.to_monitor
                ]
                return np.array(result)

            if (test_current_obj(current_objective, best_objective)).all():
                tracker.log_objective(
                    patience_counter,
                    key=(
                        "Validation",
                        "patience",
                    ),
                )
                best_objective = current_objective
                patience_counter = -1
            else:
                patience_counter += 1
                tracker.log_objective(
                    patience_counter,
                    key=(
                        "Validation",
                        "patience",
                    ),
                )
            checkpointing.save(
                epoch=epoch,
                score=current_objective[config.to_monitor[0]][
                    "eval" if config.maximize else "loss"
                ],
                patience_counter=patience_counter,
            )  # save model

            if (config.scheduler == "manual") and (
                epoch in config.scheduler_options["milestones"]
            ):
                decay_lr()

        if (epoch < max_iter) & (lr_decay_steps > 1) & (repeat < lr_decay_steps):
            if (config.scheduler == "adaptive") and (
                config.scheduler_options["mtl"]
            ):  # adaptive lr scheduling for mtl alongside early_stopping
                scheduler.step()
            decay_lr()
        patience_counter = -1

    finalize(best_objective)
