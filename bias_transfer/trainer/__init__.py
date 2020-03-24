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


def main_loop(model, criterion, device, optimizer, data_loader, epoch: int, modules, train_mode=True):
    model.train() if train_mode else model.eval()
    epoch_loss, correct, total, module_losses = 0, 0, 0, {}
    for module in modules:
        if module.criterion:  # some modules may compute an additonal output/loss
            module_losses[module.__class__.__name__] = 0
    if hasattr(tqdm, '_instances'):  # To have tqdm output without line-breaks between steps
        tqdm._instances.clear()
    with torch.enable_grad() if train_mode else torch.no_grad():
        with tqdm(data_loader, desc='{} Epoch {}'.format("Train" if train_mode else "Eval", epoch)) as t:
            for module in modules:
                module.pre_epoch(model, train_mode)
            for batch_idx, (inputs, targets) in enumerate(t):
                # Pre-Forward
                loss = torch.zeros(1, device=device)
                if train_mode:
                    optimizer.zero_grad()
                inputs, targets = inputs.to(device), targets.to(device)
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
                epoch_loss += loss.item()
                # Book-keeping
                _, predicted = outputs["logits"].max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                acc = 100. * correct / total

                def average_loss(loss_):
                    return loss_ / (batch_idx + 1)

                t.set_postfix(acc=acc, loss=average_loss(epoch_loss),
                              **{k: average_loss(l) for k, l in module_losses.items()})
                if train_mode:
                    # Backward
                    loss.backward()
                    optimizer.step()
    return acc, average_loss(epoch_loss), {k: average_loss(l) for k, l in module_losses.items()}


def test_model(model, path, criterion, device, data_loader, config, seed, noise_test: bool = True):
    model, _, epoch = load_checkpoint(path, model)
    if config.noise_test and noise_test:
        test_acc = {}
        test_loss = {}
        for n_type, n_vals in config.noise_test.items():
            test_acc[n_type] = {}
            test_loss[n_type] = {}
            for val in n_vals:
                val_str = stringify(val)
                config = copy.deepcopy(config)
                config.noise_snr = None
                config.noise_std = None
                setattr(config, n_type, val)
                main_loop_modules = [globals().get("NoiseAugmentation")(config, device, data_loader, seed)]
                test_acc[n_type][val_str], test_loss[n_type][val_str], _ = main_loop(model,
                                                                                     criterion,
                                                                                     device,
                                                                                     None,
                                                                                     data_loader=data_loader,
                                                                                     epoch=epoch,
                                                                                     modules=main_loop_modules,
                                                                                     train_mode=False)
    else:
        main_loop_modules = []
        for k in config.main_loop_modules:
            if k != "NoiseAugmentation":
                main_loop_modules.append(globals().get(k)(config, device, data_loader, seed))
        test_acc, test_loss, _ = main_loop(model,
                                           criterion,
                                           device,
                                           None,
                                           data_loader=data_loader,
                                           epoch=epoch,
                                           modules=main_loop_modules,
                                           train_mode=False)
    return test_acc, test_loss


def trainer(model, dataloaders, seed, uid, cb, eval_only=False, **kwargs):
    config = TrainerConfig.from_dict(kwargs)
    uid = nnf.utility.dj_helpers.make_hash(uid)
    device = 'cuda' if torch.cuda.is_available() and not config.force_cpu else 'cpu'
    best_acc, best_epoch = 0, 0  # best test accuracy
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


    criterion = nn.CrossEntropyLoss()
    main_loop_modules = [globals().get(k)(config, device, dataloaders["train"], seed) for k in config.main_loop_modules]

    if not eval_only:
        if config.optimizer == "Adam":
            optimizer = optim.Adam(model.parameters(),
                                   lr=config.lr,
                                   weight_decay=config.weight_decay)
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
        if config.lr_milestones:
            train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=config.lr_milestones,
                                                             gamma=config.lr_decay
                                                             )  # learning rate decay

        if config.transfer_from_path:
            model = load_model(config.transfer_from_path, model, ignore_missing=True)
            if config.reset_linear:
                model.linear_readout.reset_parameters()
            if config.freeze:
                model.freeze(config.freeze)
        else:
            path = "./checkpoint/ckpt.{}.pth".format(uid)
            if os.path.isfile(path):
                model, best_acc, start_epoch = load_checkpoint(path, model, optimizer)
                best_epoch = start_epoch

        print('==> Starting model {}'.format(config.description), flush=True)
        train_stats = []
        for epoch in range(start_epoch + 1, config.num_epochs):
            if cb:
                cb()
            train_acc, train_loss, train_module_loss = main_loop(model=model,
                                                                 criterion=criterion,
                                                                 device=device,
                                                                 optimizer=optimizer,
                                                                 data_loader=dataloaders["train"],
                                                                 epoch=epoch,
                                                                 modules=main_loop_modules,
                                                                 train_mode=True
                                                                 )
            dev_acc, dev_loss, dev_module_loss = main_loop(model=model,
                                                           criterion=criterion,
                                                           device=device,
                                                           optimizer=None,
                                                           data_loader=dataloaders["val"],
                                                           epoch=epoch,
                                                           modules=main_loop_modules,
                                                           train_mode=False
                                                           )

            if dev_acc > best_acc:
                save_checkpoint(model, optimizer, dev_acc, epoch, "./checkpoint", "ckpt.{}.pth".format(uid))
                best_acc = dev_acc
                best_epoch = best_epoch
            if config.lr_milestones:
                train_scheduler.step(epoch=epoch)
            train_stats.append({"train_acc": train_acc, "train_loss": train_loss, "train_module_loss": train_module_loss,
                                "dev_acc": dev_acc, "dev_loss": dev_loss, "dev_module_loss": dev_module_loss})
    else:
        train_stats = []

    # test the final model with noise on the dev-set
    dev_noise_acc, dev_noise_loss = test_model(model=model, path="./checkpoint/ckpt.{}.pth".format(uid),
                                               criterion=criterion, device=device, data_loader=dataloaders["val"],
                                               config=config, noise_test=True, seed=seed)
    # test the final model on the test set
    test_acc, test_loss = test_model(model=model, path="./checkpoint/ckpt.{}.pth".format(uid),
                                     criterion=criterion, device=device, data_loader=dataloaders["test"],
                                     config=config, noise_test=False, seed=seed)
    final_results = {"test_acc": test_acc,
                     "test_loss": test_loss,
                     "dev_acc": best_acc,
                     "epoch": best_epoch,
                     "dev_noise_acc": dev_noise_acc,
                     "dev_noise_loss": dev_noise_loss}

    return test_acc, (train_stats, final_results), model.state_dict()
