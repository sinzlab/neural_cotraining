from functools import partial

import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm
from neural_cotraining.models.utils import (
    freeze_params,
    reset_params,
    freeze_mtl_shared_block,
)
from neural_cotraining.trainer.utils import (
    get_subdict,
    set_bn_to_eval,
    neural_full_objective,
    move_data,
)
from .main_loop_modules import *
import sys
import nnfabrik as nnf
from nntransfer.trainer.trainer import Trainer
from nntransfer.trainer.utils.checkpointing import LocalCheckpointing
from neuralpredictors.training.tracking import AdvancedTracker
from neural_cotraining.configs.trainer import CoTrainerConfig
from neuralpredictors.measures import modules as mlmeasures
from .early_stopping import early_stopping
from nnvision.utility.measures import get_correlations, get_poisson_loss
from .utils import XEntropyLossWrapper, NBLossWrapper, MSELossWrapper
from neural_cotraining.trainer import utils as uts
from neural_cotraining.trainer import cyclers as cyclers
from nnfabrik.utility.nn_helpers import move_to_device


def trainer(model, dataloaders, seed, uid, cb, eval_only=False, **kwargs):
    t = MTLTrainer(dataloaders, model, seed, uid, cb, **kwargs)
    return t.train()


class MTLTrainer(Trainer):
    checkpointing_cls = LocalCheckpointing

    def __init__(self, dataloaders, model, seed, uid, cb, **kwargs):
        self.config = CoTrainerConfig.from_dict(kwargs)
        self.uid = uid
        self.model, self.device, self.multi = nnf.utility.nn_helpers.move_to_device(
            model, gpu=(not self.config.force_cpu)
        )
        nnf.utility.nn_helpers.set_random_seed(seed)
        self.seed = seed

        if (
            self.config.mtl
            and ("v1" not in self.config.loss_functions.keys())
            and ("v4" not in self.config.loss_functions.keys())
        ):
            if "img_classification" in dataloaders["train"].keys():
                dataloaders["train"] = (
                    dataloaders["train"]["img_classification"]
                    if isinstance(dataloaders["train"]["img_classification"], dict)
                    else get_subdict(dataloaders["train"], ["img_classification"])
                )
                dataloaders["validation"] = get_subdict(
                    dataloaders["validation"], ["img_classification"]
                )
                dataloaders["test"] = get_subdict(
                    dataloaders["test"], ["img_classification"]
                )

        self.cycler_args = dict(self.config.train_cycler_args)
        if self.cycler_args and self.config.train_cycler == "MTL_Cycler":
            self.cycler_args["ratio"] = self.cycler_args["ratio"][1]
        self.train_loader = getattr(cyclers, self.config.train_cycler)(
            dataloaders["train"], **self.cycler_args
        )
        self.val_keys = list(dataloaders["validation"].keys())
        self.data_loaders = dataloaders

        self.optimizer, self.stop_closure, self.criterion = self.get_training_controls()
        self.lr_scheduler = self.prepare_lr_schedule()

        # Potentially freeze/reset parts of the model (after loading pretrained parameters)
        self.freeze()

        # Prepare iterator for training
        print("==> Starting model {}".format(self.config.comment), flush=True)
        self.train_stats = []
        checkpointing = self.checkpointing_cls(
            self.model,
            self.optimizer,
            self.lr_scheduler,
            self.tracker,
            self.config.chkpt_options,
            self.config.maximize,
            partial(cb, uid=uid),
            hash=nnf.utility.dj_helpers.make_hash(uid),
        )
        self.epoch_iterator = early_stopping(
            self.model,
            self.stop_closure,
            self.config,
            self.optimizer,
            checkpointing=checkpointing,
            interval=self.config.interval,
            patience=self.config.patience,
            max_iter=self.config.max_iter,
            maximize=self.config.maximize,
            tolerance=self.config.threshold,
            restore_best=self.config.restore_best,
            tracker=self.tracker,
            scheduler=self.lr_scheduler,
            lr_decay_steps=self.config.lr_decay_steps,
        )

    @property
    def main_loop_modules(self):
        self._main_loop_modules = [
            globals().get(k)(
                self.model, self.config, self.device, self.train_loader, self.seed
            )
            for k in self.config.main_loop_modules
        ]
        return self._main_loop_modules

    def get_training_controls(self):
        criterion, stop_closure = {}, {}
        for k in self.val_keys:
            # in case of neural datasets
            if k in ["v1", "v4"]:
                # get the loss function with one of two cases: normal loss or likelihood loss for weighing
                if self.config.loss_weighing:
                    if self.config.loss_functions[k] == "PoissonLoss":
                        criterion[k] = NBLossWrapper(self.config.loss_sum).to(
                            self.device
                        )
                    else:
                        criterion[k] = MSELossWrapper(
                            getattr(nn, self.config.loss_functions[k])(
                                reduction="sum" if self.config.loss_sum else "mean"
                            )
                        ).to(self.device)
                else:
                    if self.config.loss_functions[k] == "PoissonLoss":
                        criterion[k] = getattr(
                            mlmeasures, self.config.loss_functions[k]
                        )(avg=self.config.avg_loss)
                    else:
                        criterion[k] = getattr(nn, self.config.loss_functions[k])(
                            reduction="sum" if self.config.loss_sum else "mean"
                        )

                # get the evaluation function to apply to the validation/test sets
                stop_closure[k] = {}
                stop_closure[k] = partial(
                    self.test_neural_model,
                    data_loader=self.data_loaders["validation"][k],
                    mode="Validation",
                    neural_set=k,
                )

            if k == "img_classification":
                if self.config.loss_weighing:
                    criterion[k] = XEntropyLossWrapper(
                        getattr(nn, self.config.loss_functions[k])(
                            reduction="sum" if self.config.loss_sum else "mean"
                        )
                    ).to(self.device)
                else:
                    criterion[k] = getattr(nn, self.config.loss_functions[k])(
                        reduction="sum" if self.config.loss_sum else "mean"
                    )
                stop_closure[k] = partial(
                    self.main_loop,
                    data_loader=self.data_loaders["validation"][k]
                    if isinstance(self.data_loaders["validation"][k], dict)
                    else get_subdict(self.data_loaders["validation"], [k]),
                    criterion=get_subdict(criterion, [k]),
                    epoch=0,
                    mode="Validation",
                    return_outputs=False,
                    cycler="LongCycler",
                    cycler_args={},
                    loss_weighing=self.config.loss_weighing,
                    mtl=self.config.mtl,
                    freeze_bn={"last_layer": -1},
                    epoch_tqdm=None,
                )

        # get all params and the optimizer instance
        if not self.config.specific_opt_options:
            all_params = list(self.model.parameters())
            if self.config.loss_weighing:
                for _, loss_object in criterion.items():
                    all_params += list(loss_object.parameters())
        else:
            params = {"params": []}
            specific_opt_options_dict = self.config.specific_opt_options
            for name, parameter in self.model.named_parameters():
                param_exist = False
                for name_part in specific_opt_options_dict.keys():
                    if name_part in name:
                        specific_opt_options_dict[name_part]["params"].append(parameter)
                        param_exist = True
                        break
                if not param_exist:
                    params["params"].append(parameter)

            if self.config.loss_weighing:
                for _, loss_object in criterion.items():
                    params["params"] += list(loss_object.parameters())
            all_params = [params] + list(specific_opt_options_dict.values())
        optimizer = getattr(optim, self.config.optimizer)(
            all_params, **self.config.optimizer_options
        )
        return optimizer, stop_closure, criterion

    def prepare_lr_schedule(self):
        if self.config.scheduler:
            if self.config.scheduler == "adaptive":
                if self.config.scheduler_options["mtl"]:
                    train_scheduler = optim.lr_scheduler.StepLR(
                        self.optimizer, step_size=1, gamma=self.config.lr_decay
                    )
                elif not self.config.scheduler_options["mtl"]:
                    train_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        self.optimizer,
                        factor=self.config.lr_decay,
                        patience=self.config.patience,
                        threshold=self.config.threshold,
                        verbose=self.config.verbose,
                        min_lr=self.config.min_lr,
                        mode="max" if self.config.maximize else "min",
                        threshold_mode=self.config.threshold_mode,
                    )
            elif self.config.scheduler == "manual":
                train_scheduler = optim.lr_scheduler.MultiStepLR(
                    self.optimizer,
                    milestones=self.config.scheduler_options["milestones"],
                    gamma=self.config.lr_decay,
                )
        else:
            train_scheduler = None
        return train_scheduler

    def freeze(self):
        if self.config.freeze:
            if self.config.mtl:
                freeze_mtl_shared_block(
                    self.model,
                    self.multi,
                    [
                        task
                        for task in list(self.config.loss_functions.keys())
                        if task in ["v1", "v4"]
                    ],
                )
                # model.freeze(config.freeze['freeze'])
            else:
                if self.config.freeze["freeze"] == ("core",):
                    kwargs = {"not_to_freeze": (self.config.readout_name,)}
                elif self.config.freeze["freeze"] == ("readout",):
                    kwargs = {"to_freeze": (self.config.readout_name,)}
                else:
                    kwargs = {"to_freeze": self.config.freeze["freeze"]}
                freeze_params(self.model, **kwargs)

            if self.config.freeze.get("reset", False):
                reset_params(self.model, self.config.freeze["reset"])

    @property
    def tracker(self):
        try:
            return self._tracker
        except AttributeError:
            objectives = {
                "LR": 0,
                "Training": {
                    task: {"loss": 0, "eval": 0, "normalization": 0}
                    for task in self.config.loss_functions
                },
                "Validation": {
                    task: {"loss": 0, "eval": 0} for task in self.config.loss_functions
                },
            }
            objectives["Validation"]["patience"] = 0
            if "img_classification" in self.config.loss_functions:
                objectives["Validation"]["img_classification"]["normalization"] = 0
            if self.config.loss_weighing:
                for task in self.config.loss_functions:
                    objectives["Training"][task]["loss_weight"] = 0
            self._tracker = AdvancedTracker(
                main_objective=(
                    "img_classification"
                    if "img_classification" in self.config.loss_functions
                    else list(self.config.loss_functions.keys())[0],
                    "eval"
                    if "img_classification" in self.config.loss_functions
                    else "loss",
                ),
                **objectives
            )
            return self._tracker

    def compute_loss(
        self,
        mode,
        data_key,
        data_loader,
        loss,
        criterion,
        loss_weighing,
        inputs,
        mtl,
        outputs,
        targets,
    ):
        if "v1" in targets.keys() or "v4" in targets.keys():
            neural_set = "v1" if "v1" in targets.keys() else "v4"
            loss += neural_full_objective(
                self.model,
                outputs[neural_set],
                data_loader,
                criterion[neural_set],
                self.config.scale_loss,
                data_key,
                inputs,
                targets[neural_set],
                multi=self.multi,
                neural_set=neural_set,
                mtl=mtl,
            )

            batch_size = targets[neural_set].size(0)
            self.tracker.log_objective(
                batch_size,
                key=(mode, neural_set, "normalization"),
            )
            self.tracker.log_objective(
                loss.item() * batch_size,
                key=(mode, neural_set, "loss"),
            )

            if loss_weighing:
                self.tracker.log_objective(
                    np.exp(criterion[neural_set].log_w.item()),
                    key=(mode, neural_set, "loss_weight"),
                )

        if "img_classification" in targets.keys():
            loss += criterion["img_classification"](
                outputs["img_classification"], targets["img_classification"]
            )
            _, predicted = outputs["img_classification"].max(1)
            self.tracker.log_objective(
                100 * predicted.eq(targets["img_classification"]).sum().item(),
                key=(mode, "img_classification", "eval"),
            )
            batch_size = targets["img_classification"].size(0)
            self.tracker.log_objective(
                batch_size,
                key=(mode, "img_classification", "normalization"),
            )
            self.tracker.log_objective(
                loss.item() * batch_size,
                key=(mode, "img_classification", "loss"),
            )

            if loss_weighing and mode == "Training":
                self.tracker.log_objective(
                    np.exp(criterion["img_classification"].log_w.item()),
                    key=(mode, "img_classification", "loss_weight"),
                )
        return loss

    def main_loop(
        self,
        data_loader,
        criterion,
        epoch: int = 0,
        mode="Training",
        return_outputs=False,
        cycler="LongCycler",
        cycler_args={},
        loss_weighing=False,
        mtl=False,
        freeze_bn={"last_layer": -1},
        epoch_tqdm=None,
    ):
        train_mode = True if mode == "Training" else False
        self.model.train() if train_mode else self.model.eval()
        if train_mode and freeze_bn["last_layer"] > 0:
            set_bn_to_eval(
                self.model,
                freeze_bn,
                self.multi,
                [task for task in list(criterion.keys()) if task in ["v1", "v4"]],
            )

        module_losses = {}
        collected_outputs = []
        tasks = {"img_classification": "labels", "v1": "responses", "v4": "responses"}
        tasks = uts.get_subdict(tasks, list(criterion.keys()))

        for module in self.main_loop_modules:
            if module.criterion:  # some modules may compute an additonal output/loss
                module_losses[module.__class__.__name__] = 0

        if cycler_args and cycler == "MTL_Cycler":
            cycler_args = dict(cycler_args)
            cycler_args["ratio"] = cycler_args["ratio"][
                max(i for i in list(cycler_args["ratio"].keys()) if epoch >= i)
            ]
        data_cycler = getattr(cyclers, cycler)(data_loader, **cycler_args)
        n_iterations = len(data_cycler)
        if hasattr(
            tqdm, "_instances"
        ):  # To have tqdm output without line-breaks between steps
            tqdm._instances.clear()
        with torch.enable_grad() if train_mode else torch.no_grad():

            with tqdm(
                enumerate(data_cycler),
                total=n_iterations,
                desc="{} Epoch {}".format(mode, epoch),
                disable=self.config.show_epoch_progress,
                file=sys.stdout,
            ) as t:

                for module in self.main_loop_modules:
                    module.pre_epoch(
                        self.model, train_mode, epoch, optimizer=self.optimizer
                    )

                if train_mode:
                    self.optimizer.zero_grad()

                for batch_idx, batch_data in t:
                    # Pre-Forward
                    loss = torch.zeros(1, device=self.device)
                    inputs, targets, data_key, neural_set = move_data(
                        batch_data, tasks, self.device
                    )
                    shared_memory = {}  # e.g. to remember where which noise was applied
                    model_ = self.model
                    for module in self.main_loop_modules:
                        model_, inputs = module.pre_forward(
                            model_,
                            inputs,
                            shared_memory,
                            train_mode=train_mode,
                            data_key=data_key,
                            neural_set=neural_set,
                            task_keys=list(targets.keys()),
                        )
                    # Forward
                    outputs = model_(inputs)

                    # Post-Forward and Book-keeping
                    if return_outputs:
                        collected_outputs.append(outputs[0])
                    for module in self.main_loop_modules:
                        outputs, loss, targets = module.post_forward(
                            outputs,
                            loss,
                            targets,
                            module_losses,
                            train_mode,
                            task_keys=list(targets.keys()),
                            **shared_memory
                        )

                    loss = self.compute_loss(
                        mode,
                        data_key,
                        data_loader,
                        loss,
                        criterion,
                        loss_weighing,
                        inputs,
                        mtl,
                        outputs,
                        targets,
                    )

                    self.tracker.display_log(tqdm_iterator=t, key=(mode,))

                    if train_mode:
                        # Backward
                        loss.backward()
                        for module in self.main_loop_modules:
                            module.post_backward(self.model)
                        if data_cycler.backward:
                            self.optimizer.step()
                            self.optimizer.zero_grad()

        objective = self.tracker.get_current_main_objective(
            "Training" if train_mode else "Validation"
        )
        if return_outputs:
            return (objective, collected_outputs)

        return objective

    def train(self):
        # train over epochs
        epoch = 0
        if hasattr(
            tqdm, "_instances"
        ):  # To have tqdm output without line-breaks between steps
            tqdm._instances.clear()
        with tqdm(
            iterable=self.epoch_iterator,
            total=self.config.max_iter,
            disable=(not self.config.show_epoch_progress),
        ) as epoch_tqdm:
            for epoch, dev_eval in epoch_tqdm:
                self.tracker.start_epoch()
                self.tracker.log_objective(
                    self.optimizer.param_groups[0]["lr"], ("LR",)
                )
                self.main_loop(
                    data_loader=self.data_loaders["train"],
                    mode="Training",
                    epoch=epoch,
                    epoch_tqdm=epoch_tqdm,
                    criterion=self.criterion,
                    return_outputs=False,
                    cycler=self.config.train_cycler,
                    cycler_args=self.config.train_cycler_args,
                    loss_weighing=self.config.loss_weighing,
                    mtl=self.config.mtl,
                    freeze_bn=self.config.freeze_bn,
                )

        test_result = self.test_final_model(epoch)
        return (
            test_result,
            self.tracker.state_dict(),
            self.model.state_dict(),
        )

    def test_neural_model(self, data_loader, neural_set, mode):

        objectives = {
            mode: {
                neural_set: {
                    "eval": 0,
                    "loss": 0,
                }
            }
        }

        self.tracker.add_objectives(objectives, init_epoch=True)
        loss = get_poisson_loss(
            self.model,
            data_loader,
            device=self.device,
            as_dict=False,
            per_neuron=False,
            neural_set=neural_set,
            mtl=self.config.mtl,
        )
        eval = get_correlations(
            self.model,
            data_loader,
            device=self.device,
            as_dict=False,
            per_neuron=False,
            neural_set=neural_set,
            mtl=self.config.mtl,
        )

        self.tracker.log_objective(
            loss.item(),
            key=(mode, neural_set, "loss"),
        )

        self.tracker.log_objective(
            eval.item(),
            key=(mode, neural_set, "eval"),
        )

        n_log = self.tracker._normalize_log(self.tracker.log)
        current_log = self.tracker._gather_log(n_log, (mode,), index=-1)
        print("{} {}".format(mode, current_log))
        return self.tracker.get_current_objective((mode, neural_set, "eval"))

    def test_cls_model(self, epoch, criterion, data_loader, objectives, mode):
        self.tracker.add_objectives(objectives, init_epoch=True)
        result = self.main_loop(
            data_loader=data_loader,
            criterion=criterion,
            epoch=epoch,
            mode=mode,
            cycler="LongCycler",
            cycler_args={},
            loss_weighing=self.config.loss_weighing,
            mtl=self.config.mtl,
            freeze_bn={"last_layer": -1},
            epoch_tqdm=None,
        )
        return result

    def create_cls_objective_dict(self, mode, task):
        return {
            mode: {
                task: {
                    "eval": 0,
                    "loss": 0,
                    "normalization": 0,
                }
            }
        }

    def test_final_model(self, epoch):

        self.test_result = {}
        for k in self.val_keys:
            if k in ["v1", "v4"]:
                if self.config.add_final_train_eval:
                    self.test_neural_model(
                        self.data_loaders["train"][k], k, "FinalTraining"
                    )

                if self.config.add_final_val_eval:
                    self.test_neural_model(
                        self.data_loaders["validation"][k], k, "FinalValidation"
                    )

                if self.config.add_final_test_eval:
                    self.test_result[k] = self.test_neural_model(
                        self.data_loaders["test"][k], k, "FinalTest"
                    )

            if k == "img_classification":
                if k in self.data_loaders["train"].keys():
                    if isinstance(self.data_loaders["train"][k], dict):
                        imgcls_train_loader_input = self.data_loaders["train"][k]
                    else:
                        imgcls_train_loader_input = get_subdict(
                            self.data_loaders["train"], [k]
                        )
                else:
                    imgcls_train_loader_input = self.data_loaders["train"]
                if self.config.add_final_train_eval:
                    objectives = self.create_cls_objective_dict("FinalTraining", k)
                    self.test_cls_model(
                        epoch,
                        get_subdict(self.criterion, [k]),
                        imgcls_train_loader_input,
                        objectives,
                        "FinalTraining",
                    )

                if self.config.add_final_val_eval:
                    objectives = self.create_cls_objective_dict(
                        "FinalValidationInDomain", k
                    )
                    self.test_cls_model(
                        epoch,
                        get_subdict(self.criterion, [k]),
                        self.data_loaders["validation"][k]
                        if isinstance(self.data_loaders["validation"][k], dict)
                        else get_subdict(self.data_loaders["validation"], [k]),
                        objectives,
                        "FinalValidationInDomain",
                    )

                    if "validation_out_domain" in self.data_loaders.keys():
                        objectives = self.create_cls_objective_dict(
                            "FinalValidationOutDomain", k
                        )
                        self.test_cls_model(
                            epoch,
                            get_subdict(self.criterion, [k]),
                            get_subdict(
                                self.data_loaders["validation_out_domain"], [k]
                            ),
                            objectives,
                            "FinalValidationOutDomain",
                        )

                if self.config.add_final_test_eval:
                    objectives = self.create_cls_objective_dict("FinalTestInDomain", k)
                    self.test_result[k] = self.test_cls_model(
                        epoch,
                        get_subdict(self.criterion, [k]),
                        self.data_loaders["test"][k]
                        if isinstance(self.data_loaders["test"][k], dict)
                        else get_subdict(self.data_loaders["test"], [k]),
                        objectives,
                        "FinalTestInDomain",
                    )

                    if "test_out_domain" in self.data_loaders.keys():
                        objectives = self.create_cls_objective_dict(
                            "FinalTestOutDomain", k
                        )
                        self.test_cls_model(
                            epoch,
                            get_subdict(self.criterion, [k]),
                            get_subdict(self.data_loaders["test_out_domain"], [k]),
                            objectives,
                            "FinalTestOutDomain",
                        )

        if "validation_gauss" in self.data_loaders:
            for level, dataloader in self.data_loaders["validation_gauss"].items():
                objectives = self.create_cls_objective_dict(
                    "Validation_gauss " + str(level), k
                )
                self.test_cls_model(
                    epoch,
                    get_subdict(self.criterion, ["img_classification"]),
                    {"img_classification": dataloader},
                    objectives,
                    "Validation_gauss " + str(level),
                )

        if "c_test" in self.data_loaders:
            for c_category in list(self.data_loaders["c_test"].keys()):
                for c_level, dataloader in self.data_loaders["c_test"][
                    c_category
                ].items():
                    objectives = self.create_cls_objective_dict(
                        "c_test {} {}".format(str(c_category), str(c_level)), k
                    )
                    self.test_cls_model(
                        epoch,
                        get_subdict(self.criterion, ["img_classification"]),
                        {"img_classification": dataloader},
                        objectives,
                        "c_test {} {}".format(str(c_category), str(c_level)),
                    )

        if "fly_c_test" in self.data_loaders:
            for fly_noise_type in list(self.data_loaders["fly_c_test"].keys()):
                for level, dataloader in self.data_loaders["fly_c_test"][
                    fly_noise_type
                ].items():
                    objectives = self.create_cls_objective_dict(
                        "fly_c_test {} {}".format(str(fly_noise_type), str(level)), k
                    )
                    self.test_cls_model(
                        epoch,
                        get_subdict(self.criterion, ["img_classification"]),
                        {"img_classification": dataloader},
                        objectives,
                        "fly_c_test {} {}".format(str(fly_noise_type), str(level)),
                    )

        if "imagenet_fly_c_test" in self.data_loaders:
            for fly_noise_type in list(self.data_loaders["imagenet_fly_c_test"].keys()):
                for level, dataloader in self.data_loaders["imagenet_fly_c_test"][
                    fly_noise_type
                ].items():
                    objectives = self.create_cls_objective_dict(
                        "imagenet_fly_c_test {} {}".format(
                            str(fly_noise_type), str(level)
                        ),
                        k,
                    )
                    self.test_cls_model(
                        epoch,
                        get_subdict(self.criterion, ["img_classification"]),
                        {"img_classification": dataloader},
                        objectives,
                        "imagenet_fly_c_test {} {}".format(
                            str(fly_noise_type), str(level)
                        ),
                    )
        return list(self.test_result.values())[0]
