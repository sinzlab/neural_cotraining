import os
from functools import partial

import numpy as np
import torch
from torch import nn, optim
from torch.backends import cudnn as cudnn

import nnfabrik as nnf
from bias_transfer.configs.trainer import TrainerConfig
from bias_transfer.trainer import main_loop
from bias_transfer.trainer.transfer import transfer_model
from bias_transfer.trainer.main_loop import main_loop
from bias_transfer.trainer.test import test_neural_model, test_model
from bias_transfer.utils.io import load_model, load_checkpoint, save_checkpoint
from mlutils import measures as mlmeasures
from mlutils.training import LongCycler, MultipleObjectiveTracker, early_stopping
from nnvision.utility import measures
from nnvision.utility.measures import get_correlations, get_poisson_loss
from bias_transfer.trainer.main_loop_modules import *


def trainer(model, dataloaders, seed, uid, cb, eval_only=False, **kwargs):
    config = TrainerConfig.from_dict(kwargs)
    uid = nnf.utility.dj_helpers.make_hash(uid)
    device = "cuda" if torch.cuda.is_available() and not config.force_cpu else "cpu"
    best_eval, best_epoch = -1000000, 0  # best test eval
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Model
    print("==> Building model..", flush=True)
    model = model.to(device)
    if device == "cuda":
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)

    train_loader = (
        LongCycler(dataloaders["train"])
        if config.neural_prediction
        else dataloaders["train"]
    )
    train_n_iterations = len(train_loader)
    val_n_iterations = len(
        LongCycler(dataloaders["validation"])
        if config.neural_prediction
        else dataloaders["validation"]
    )
    test_n_iterations = 1 if config.neural_prediction else len(dataloaders["test"])

    # Main-loop modules:
    main_loop_modules = [
        globals().get(k)(model, config, device, train_loader, seed)
        for k in config.main_loop_modules
    ]

    if config.neural_prediction:
        criterion = getattr(mlmeasures, config.loss_function)(avg=config.avg_loss)
        stop_closure = partial(
            getattr(measures, config.stop_function),
            dataloaders=dataloaders["validation"],
            device=device,
            per_neuron=False,
            avg=True,
        )
        # set the number of iterations over which you would like to accumulate gradients
        optim_step_count = (
            len(dataloaders["train"].keys())
            if config.loss_evalum_batch_n is None
            else config.loss_evalum_batch_n
        )
    else:
        criterion = getattr(nn, config.loss_function)()
        optim_step_count = 1
        stop_closure = partial(
            main_loop,
            criterion=criterion,
            device=device,
            data_loader=dataloaders["validation"],
            modules=main_loop_modules,
            train_mode=False,
            n_iterations=val_n_iterations,
            return_outputs=False,
            neural_prediction=config.neural_prediction,
            scale_loss=True,
            optim_step_count=optim_step_count,
            eval_type="Validation",
            return_eval=True,
            epoch=0,
            optimizer=None,
        )


    if config.track_training:
        tracker_dict = dict(
            correlation=partial(
                get_correlations(),
                model,
                dataloaders["validation"],
                device=device,
                per_neuron=False,
            ),
            poisson_loss=partial(
                get_poisson_loss(),
                model,
                dataloaders["validation"],
                device=device,
                per_neuron=False,
                avg=False,
            ),
        )
        if hasattr(model, "tracked_values"):
            tracker_dict.update(model.tracked_values)
        tracker = MultipleObjectiveTracker(**tracker_dict)
    else:
        tracker = None

    optimizer = getattr(optim, config.optimizer)(
        model.parameters(), **config.optimizer_options
    )
    if config.adaptive_lr:
        train_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config.lr_decay,
            patience=config.patience,
            threshold=config.threshold,
            verbose=config.verbose,
            min_lr=config.min_lr,
            mode="max" if config.maximize else "min",
            threshold_mode=config.threshold_mode,
        )
    elif config.lr_milestones:
        train_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=config.lr_milestones, gamma=config.lr_decay
        )  # learning rate decay

    start_epoch = config.epoch
    path = "./checkpoint/ckpt.{}.pth".format(uid)
    if os.path.isfile(path):
        model, best_eval, start_epoch = load_checkpoint(path, model, optimizer)
        best_epoch = start_epoch
    elif config.transfer_from_path:
        dataloaders["train"] = transfer_model(
            model,
            config,
            criterion=criterion,
            device=device,
            data_loader=dataloaders["train"],
        )

    print("==> Starting model {}".format(config.comment), flush=True)
    train_stats = []

    # train over epochs
    for epoch, dev_eval in early_stopping(
        model,
        stop_closure,
        interval=config.interval,
        patience=config.patience,
        start=start_epoch,
        max_iter=config.max_iter,
        maximize=config.maximize,
        tolerance=config.threshold,
        restore_best=config.restore_best,
        tracker=tracker,
        scheduler=train_scheduler if config.adaptive_lr else None,
        lr_decay_steps=config.lr_decay_steps,
    ):
        if cb:
            cb()

        if config.verbose and tracker is not None:
            print("=======================================")
            for key in tracker.log.keys():
                print(key, tracker.log[key][-1], flush=True)

        train_eval, train_loss, train_module_loss = main_loop(
            model=model,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            data_loader=dataloaders["train"],
            n_iterations=train_n_iterations,
            modules=main_loop_modules,
            train_mode=True,
            epoch=epoch,
            neural_prediction=config.neural_prediction,
            optim_step_count=optim_step_count,
        )
        if dev_eval > best_eval:
            save_checkpoint(
                model,
                optimizer,
                dev_eval,
                epoch,
                "./checkpoint",
                "ckpt.{}.pth".format(uid),
            )
            best_eval = dev_eval
            best_epoch = epoch
        if config.lr_milestones:  # TODO: see if still working correctly
            train_scheduler.step(epoch=epoch)

        train_stats.append(
            {
                "train_eval": train_eval,
                "train_loss": train_loss,
                "train_module_loss": train_module_loss,
                "dev_eval": dev_eval,
            }
        )

    if not config.lottery_ticket:
        model, _, epoch = load_checkpoint("./checkpoint/ckpt.{}.pth".format(uid), model)
    else:
        for module in main_loop_modules:
            module.pre_epoch(model, True, epoch+1)

    # test the final model with noise on the dev-set
    dev_noise_eval, dev_noise_loss = test_model(
        model=model,
        epoch=epoch,
        n_iterations=val_n_iterations,
        criterion=criterion,
        device=device,
        data_loader=dataloaders["validation"],
        config=config,
        noise_test=True,
        seed=seed,
    )
    # test the final model on the test set
    if config.neural_prediction:
        test_eval, test_loss = test_neural_model(
            model, data_loader=dataloaders["test"], device=device, epoch=epoch
        )
    else:
        test_eval, test_loss = test_model(
            model=model,
            epoch=epoch,
            n_iterations=test_n_iterations,
            criterion=criterion,
            device=device,
            data_loader=dataloaders["test"],
            config=config,
            noise_test=False,
            seed=seed,
            eval_type="Test",
        )

    final_results = {
        "test_eval": test_eval,
        "test_loss": test_loss,
        "dev_eval": best_eval,
        "epoch": best_epoch,
        "dev_noise_eval": dev_noise_eval,
        "dev_noise_loss": dev_noise_loss,
    }

    if "c_test" in dataloaders:
        c_test_eval, c_test_loss = {}, {}
        for c_category in dataloaders["c_test"].keys():
            c_test_eval[c_category], c_test_loss[c_category] = {}, {}
            for c_level, dataloader in dataloaders["c_test"][c_category].items():
                eval, loss = test_model(
                    model=model,
                    n_iterations=len(dataloader),
                    epoch=epoch,
                    criterion=criterion,
                    device=device,
                    data_loader=dataloader,
                    config=config,
                    noise_test=False,
                    seed=seed,
                    eval_type="Test-C",
                )
                c_test_eval[c_category][c_level] = eval
                c_test_loss[c_category][c_level] = loss
        final_results["c_test_eval"] = c_test_eval
        final_results["c_test_loss"] = c_test_loss

    return test_eval, (train_stats, final_results), model.state_dict()
