import copy
import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
import nnfabrik as nnf

import os
from tqdm import tqdm

from bias_transfer.utils import stringify
from bias_transfer.utils.io import *
from .representation_matching import RepresentationMatching
from .noise_augmentation import NoiseAugmentation
from .noise_adv_training import NoiseAdvTraining
from .random_readout_reset import RandomReadoutReset
import numpy as np

from bias_transfer.configs.trainer import TrainerConfig

import os

from nnfabrik.utility.nn_helpers import get_module_output

from functools import partial

from tqdm import tqdm

from mlutils.measures import *
from mlutils import measures as mlmeasures
from mlutils.training import early_stopping, MultipleObjectiveTracker, LongCycler

from nnvision.utility import measures
from nnvision.utility.measures import get_correlations, get_poisson_loss


import numpy as np

from torch import nn



def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


def neural_full_objective(model, outputs, dataloader, criterion, device, scale_loss, data_key, inputs, targets):
    loss_scale = np.sqrt(len(dataloader[data_key].dataset) / inputs.shape[0]) if scale_loss else 1.0

    return loss_scale * criterion(outputs, targets.to(device)) + model.regularizer(data_key)


def main_loop(model,
              criterion,
              device,
              optimizer,
              data_loader,
              epoch: int,
              n_iterations,
              modules,
              train_mode=True,
              return_outputs=False,
              neural_prediction=False,
              scale_loss=True,
              optim_step_count=1):
    model.train() if train_mode else model.eval()
    epoch_loss, correct, total, module_losses, collected_outputs = 0, 0, 0, {}, []
    for module in modules:
        if module.criterion:  # some modules may compute an additonal output/loss
            module_losses[module.__class__.__name__] = 0
    if hasattr(tqdm, '_instances'):  # To have tqdm output without line-breaks between steps
        tqdm._instances.clear()
    with torch.enable_grad() if train_mode else torch.no_grad():
        with tqdm(enumerate(LongCycler(data_loader) if neural_prediction else data_loader), total=n_iterations,
                  desc='{} Epoch {}'.format("Train" if train_mode else "Eval", epoch)) as t:
            for module in modules:
                module.pre_epoch(model, train_mode)
            for batch_idx, batch_data in t:
                # Pre-Forward
                loss = torch.zeros(1, device=device)
                if train_mode:
                    optimizer.zero_grad()
                if neural_prediction:
                    data_key, (inputs, targets) = batch_data[0], batch_data[1]
                    batch_dict = {data_key: [(inputs,targets)] }
                    inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device)
                else:
                    inputs, targets = batch_data[0].to(device,dtype=torch.float), batch_data[1].to(device)
                shared_memory = {}  # e.g. to remember where which noise was applied
                for module in modules:
                    model, inputs = module.pre_forward(model, inputs, shared_memory, train_mode=train_mode)
                # Forward
                if neural_prediction:
                    outputs = model(inputs, data_key)
                else:
                    outputs = model(inputs)
                # Post-Forward
                for module in modules:
                    outputs, loss = module.post_forward(outputs, loss, module_losses, train_mode=train_mode,
                                                        **shared_memory)
                if return_outputs:
                    collected_outputs.append(outputs)
                if neural_prediction:
                    loss += neural_full_objective(model, outputs, data_loader, criterion,
                                                  device, scale_loss, data_key,
                                                  inputs, targets)
                else:
                    loss += criterion(outputs["logits"], targets)
                epoch_loss += loss.item()
                # Book-keeping
                def average_loss(loss_):
                    return loss_ / (batch_idx + 1)

                if neural_prediction:
                    total += get_correlations(model, batch_dict, device=device,
                                                              as_dict=False, per_neuron=False)
                    eval = average_loss(total)
                else:
                    _, predicted = outputs["logits"].max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    eval = 100. * correct / total

                t.set_postfix(eval=eval, loss=average_loss(epoch_loss),
                              **{k: average_loss(l) for k, l in module_losses.items()})
                if train_mode:
                    # Backward
                    loss.backward()
                    if (batch_idx + 1) % optim_step_count == 0:
                        optimizer.step()
                        optimizer.zero_grad()

    if return_outputs:
        return eval, average_loss(epoch_loss), {k: average_loss(l) for k, l in module_losses.items()}, collected_outputs
    return eval, average_loss(epoch_loss), {k: average_loss(l) for k, l in module_losses.items()}


def test_model(model, epoch, n_iterations, criterion, device, data_loader, config, seed, noise_test: bool = True):
    if config.noise_test and noise_test:
        test_eval = {}
        test_loss = {}
        for n_type, n_vals in config.noise_test.items():
            test_eval[n_type] = {}
            test_loss[n_type] = {}
            for val in n_vals:
                val_str = stringify(val)
                config = copy.deepcopy(config)
                config.noise_snr = None
                config.noise_std = None
                setattr(config, n_type, val)
                main_loop_modules = [globals().get("NoiseAugmentation")(config, device, data_loader, seed)]
                test_eval[n_type][val_str], test_loss[n_type][val_str], _ = main_loop(model,
                                                                                     criterion,
                                                                                     device,
                                                                                     None,
                                                                                     data_loader=data_loader,
                                                                                     epoch=epoch, n_iterations=n_iterations,
                                                                                     modules=main_loop_modules,
                                                                                     train_mode=False,
                                                                                      neural_prediction=config.neural_prediction)
    else:
        main_loop_modules = []
        for k in config.main_loop_modules:
            if k != "NoiseAugmentation":
                main_loop_modules.append(globals().get(k)(config, device, data_loader, seed))
        test_eval, test_loss, _ = main_loop(model,
                                           criterion,
                                           device,
                                           None,
                                           data_loader=data_loader,
                                           epoch=epoch,n_iterations=n_iterations,
                                           modules=main_loop_modules,
                                           train_mode=False,
                                            neural_prediction=config.neural_prediction)
    return test_eval, test_loss

def img_cls_stop_func(model,
              criterion,
              device,
              data_loader,
              modules, train_mode=False):
    model.eval()
    correct, total, module_losses = 0, 0, {}
    for module in modules:
        if module.criterion:  # some modules may compute an additonal output/loss
            module_losses[module.__class__.__name__] = 0

    with torch.no_grad():
            for module in modules:
                module.pre_epoch(model, train_mode)
            for batch_idx, batch_data in enumerate(data_loader):
                # Pre-Forward
                loss = torch.zeros(1, device=device)
                inputs, targets = batch_data[0].to(device,dtype=torch.float), batch_data[1].to(device)
                shared_memory = {}  # e.g. to remember where which noise was applied
                for module in modules:
                    model, inputs = module.pre_forward(model, inputs, shared_memory, train_mode=train_mode)
                # Forward
                outputs = model(inputs)
                # Post-Forward
                for module in modules:
                    outputs, loss = module.post_forward(outputs, loss, module_losses, train_mode=train_mode,
                                                        **shared_memory)
                loss += criterion(outputs["logits"], targets)
                # Book-keeping
                _, predicted = outputs["logits"].max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                eval = 100. * correct / total

    return eval


def trainer(model, dataloaders, seed, uid, cb, eval_only=False, **kwargs):
    config = TrainerConfig.from_dict(kwargs)
    uid = nnf.utility.dj_helpers.make_hash(uid)
    device = 'cuda' if torch.cuda.is_available() and not config.force_cpu else 'cpu'
    best_eval, best_epoch = 0, 0  # best test eval
    start_epoch = -1  # start from epoch 0 or last checkpoint epoch
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Model
    print('==> Building model..', flush=True)
    model = model.to(device)
    if device == 'cuda':
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)

    train_n_iterations = len(LongCycler(dataloaders['train']) if config.neural_prediction else dataloaders["train"])
    val_n_iterations = len(LongCycler(dataloaders['validation']) if config.neural_prediction else dataloaders["validation"])
    test_n_iterations = len(LongCycler(dataloaders['test']) if config.neural_prediction else dataloaders["test"])

    if config.neural_prediction:
        main_loop_modules = [globals().get(k)(config, device, LongCycler(dataloaders['train']), seed) for k in
                             config.main_loop_modules]
    else:
        main_loop_modules = [globals().get(k)(config, device, dataloaders["train"], seed) for k in config.main_loop_modules]

    if config.neural_prediction:
        criterion = getattr(mlmeasures, config.loss_function)(avg=config.avg_loss)
        stop_closure = partial(getattr(measures, config.stop_function), dataloaders=dataloaders["validation"],
                               device=device,
                               per_neuron=False, avg=True)
        # set the number of iterations over which you would like to evalummulate gradients
        optim_step_count = len(dataloaders["train"].keys()) if config.loss_evalum_batch_n is None else config.loss_evalum_batch_n
    else:
        criterion = nn.CrossEntropyLoss()
        stop_closure = partial(img_cls_stop_func, criterion=criterion, device=device,
                                data_loader=dataloaders['validation'], modules=main_loop_modules, train_mode=False)
        optim_step_count = 1

    if not eval_only:

        if config.track_training:
            tracker_dict = dict(
                correlation=partial(get_correlations(), model, dataloaders["validation"], device=device,
                                    per_neuron=False),
                poisson_loss=partial(get_poisson_loss(), model, dataloaders["validation"], device=device,
                                     per_neuron=False,
                                     avg=False))
            if hasattr(model, 'tracked_values'):
                tracker_dict.update(model.tracked_values)
            tracker = MultipleObjectiveTracker(**tracker_dict)
        else:
            tracker = None

        if config.optimizer == "Adam":
            optimizer = optim.Adam(model.parameters(),
                                   lr=config.lr,
                                   weight_decay=config.weight_decay,
                                   amsgrad=config.amsgrad)
        elif config.optimizer == "RMSprop":
            optimizer = optim.RMSprop(model.parameters(),
                                      lr=config.lr,
                                      momentum=config.momentum,
                                      weight_decay=config.weight_decay)
        else:
            optimizer = optim.SGD(model.parameters(),
                                  lr=config.lr,
                                  momentum=config.momentum,
                                  weight_decay=config.weight_decay)
        if config.adaptive_lr:
            train_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                   factor=config.lr_decay, patience=config.patience,
                                                                   threshold=config.threshold,
                                                                   verbose=config.verbose, min_lr=config.min_lr,
                                                                   mode='max' if config.maximize else 'min',
                                                                   threshold_mode=config.threshold_mode)
        elif config.lr_milestones:
            train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=config.lr_milestones,
                                                             gamma=config.lr_decay
                                                             )  # learning rate decay

        if config.transfer_from_path:
            model = load_model(config.transfer_from_path, model, ignore_missing=True)
            if config.reset_linear:
                model.readout.apply(weight_reset)
            if config.freeze:
                model.freeze(config.freeze)
        else:
            path = "./checkpoint/ckpt.{}.pth".format(uid)
            if os.path.isfile(path):
                model, best_eval, start_epoch = load_checkpoint(path, model, optimizer)
                best_epoch = start_epoch

        print('==> Starting model {}'.format(config.comment), flush=True)
        train_stats = []

        # train over epochs
        for epoch, val_obj in early_stopping(model, stop_closure, interval=config.interval, patience=config.patience,
                                             start=config.epoch, max_iter=config.max_iter, maximize=config.maximize,
                                             tolerance=config.threshold, restore_best=config.restore_best, tracker=tracker,
                                             scheduler=train_scheduler, lr_decay_steps=config.lr_decay_steps):
            if cb:
                cb()

            if config.verbose and tracker is not None:
                for key in tracker.log.keys():
                    print(key, tracker.log[key][-1], flush=True)

            train_eval, train_loss, train_module_loss = main_loop(model=model,
                                                                 criterion=criterion,
                                                                 device=device,
                                                                 optimizer=optimizer,
                                                                 data_loader=dataloaders["train"],
                                                                 epoch=epoch,
                                                                 n_iterations=train_n_iterations,
                                                                 modules=main_loop_modules,
                                                                 train_mode=True,
                                                                 neural_prediction=config.neural_prediction,
                                                                  optim_step_count=optim_step_count
                                                                 )
            dev_eval, dev_loss, dev_module_loss = main_loop(model=model,
                                                           criterion=criterion,
                                                           device=device,
                                                           optimizer=None,
                                                           data_loader=dataloaders['validation'] if config.neural_prediction else dataloaders["validation"],
                                                           epoch=epoch,
                                                           n_iterations=val_n_iterations,
                                                           modules=main_loop_modules,
                                                           train_mode=False,
                                                           neural_prediction=config.neural_prediction,
                                                           )

            if dev_eval > best_eval:
                save_checkpoint(model, optimizer, dev_eval, epoch, "./checkpoint", "ckpt.{}.pth".format(uid))
                best_eval = dev_eval
                best_epoch = epoch
            #if config.lr_milestones:
            #    train_scheduler.step(epoch=epoch)
            #elif config.adaptive_lr:
            #    train_scheduler.step(dev_loss)
            train_stats.append(
                {"train_eval": train_eval, "train_loss": train_loss, "train_module_loss": train_module_loss,
                 "dev_eval": dev_eval, "dev_loss": dev_loss, "dev_module_loss": dev_module_loss})
    else:
        train_stats = []

    model, _, epoch = load_checkpoint("./checkpoint/ckpt.{}.pth".format(uid), model)
    # test the final model with noise on the dev-set
    dev_noise_eval, dev_noise_loss = test_model(model=model, epoch=epoch, n_iterations=val_n_iterations,
                                               criterion=criterion, device=device,
                                                data_loader=dataloaders['validation'] if config.neural_prediction else dataloaders["validation"],
                                               config=config, noise_test=True, seed=seed)
    # test the final model on the test set
    test_eval, test_loss = test_model(model=model, epoch=epoch, n_iterations=test_n_iterations,
                                     criterion=criterion, device=device, data_loader=dataloaders["test"],
                                     config=config, noise_test=False, seed=seed)
    final_results = {"test_eval": test_eval,
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
                eval, loss = test_model(model=model, epoch=epoch,
                                                     criterion=criterion, device=device,
                                                     data_loader=dataloader,
                                                     config=config, noise_test=False, seed=seed)
                c_test_eval[c_category][c_level] = eval
                c_test_loss[c_category][c_level] = loss
        final_results["c_test_eval"] = c_test_eval
        final_results["c_test_loss"] = c_test_loss

    return test_eval, (train_stats, final_results), model.state_dict()
