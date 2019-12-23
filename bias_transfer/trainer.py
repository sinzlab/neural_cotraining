import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn

import datajoint as dj

import os
from tqdm import tqdm

from bias_transfer.utils import stringify


def load_model(model, path):
    print('==> Loading checkpoint..', flush=True)
    assert os.path.isfile(path), 'Error: no checkpoint file found!'
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    return model, best_acc, start_epoch


def save_model(model, acc, epoch, path, name):
    print('==> Saving..', flush=True)
    state = {
        'net': model.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir(path):
        os.mkdir(path)
    torch.save(state, os.path.join(path, name))


def apply_noise(x, device, std: dict = None, snr: dict = None):
    with torch.no_grad():
        noise_levels = std if std else snr
        assert sum(noise_levels.values()) == 1.0, "Percentage for noise levels should sum to one!"
        indices = torch.randperm(x.shape[0])
        start = 0
        for level, percentage in noise_levels.items():
            end = start + int(percentage * x.shape[0])
            if level is not None:  # option to deactivate noise for a fraction of the data
                if std is None:  # are we doing snr or std?
                    signal = torch.mean(x[indices[start:end]] * x[indices[start:end]], dim=[1, 2, 3], keepdim=True)  # for each dimension except batch
                    std = signal / level
                else:
                    std = torch.tensor(level)
                std = std.expand_as(x[start:end])
                x[indices[start:end]] += torch.normal(mean=0.0, std=std).to(device)
            start = end
        x = torch.clamp(x, max=1.0, min=0.0)
    return x


def train_loop(model, criterion, device, optimizer, data_loader, epoch, add_noise: bool = False,
               noise_std: dict = None, noise_snr: dict = None):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    if hasattr(tqdm, '_instances'):
        tqdm._instances.clear()
    with tqdm(data_loader, desc='Train Epoch {}'.format(epoch)) as t:
        for batch_idx, (inputs, targets) in enumerate(t):
            inputs, targets = inputs.to(device), targets.to(device)
            if add_noise:
                inputs = apply_noise(inputs, device, std=noise_std, snr=noise_snr)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc = 100. * correct / total
            t.set_postfix(acc=acc, loss=train_loss / total)
    return acc, train_loss / total


def test_loop(model, criterion, device, data_loader, epoch, best_acc, comment: str = "",
              add_noise: bool = False, noise_std: float = None, noise_snr: float = None, compute_corr: bool = False):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(data_loader, desc='Eval Epoch {}'.format(epoch)) as t:
            if add_noise:
                torch.manual_seed(42)  # so that we always have the same noise for evaluation!
            for batch_idx, (inputs, targets) in enumerate(t):
                inputs, targets = inputs.to(device), targets.to(device)
                if add_noise:
                    inputs = apply_noise(inputs, device, std=noise_std, snr=noise_snr)
                outputs, corr_matrices = model(inputs, compute_corr=compute_corr)
                if compute_corr:
                    return corr_matrices
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                t.set_postfix(acc=100. * correct / total, loss=loss.item())

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        save_model(model, acc, epoch, "./checkpoint", "ckpt.{}.pth".format(comment))
        best_acc = acc
    return acc, test_loss, best_acc


def test_model(model, path, criterion, device, data_loader, noise_test=None):
    model, _, _ = load_model(model, path)
    if noise_test:
        test_acc = {}
        test_loss = {}
        for noise_type, noise_vals in noise_test.items():
            test_acc[noise_type] = {}
            test_loss[noise_type] = {}
            for val in noise_vals:
                val_string = stringify(val)
                test_acc[noise_type][val_string], test_loss[noise_type][val_string], _ = test_loop(model, criterion,
                                                                                                   device,
                                                                                                   data_loader=data_loader,
                                                                                                   epoch=999,
                                                                                                   best_acc=0,
                                                                                                   comment="Final Eval",
                                                                                                   add_noise=True,
                                                                                                   **{noise_type: val})
    else:
        test_acc, test_loss, _ = test_loop(model, criterion, device,
                                           data_loader=data_loader, epoch=999,
                                           best_acc=0,
                                           comment="Final Eval",
                                           add_noise=False)
    return test_acc, test_loss


def trainer(model, seed, dataloaders,
            add_noise: bool = False,
            noise_std: dict = None,
            noise_snr: dict = None,
            noise_test: dict = None,
            num_epochs: int = 200,
            force_cpu: bool = False,
            lr: float = 0.1,
            momentum: float = 0.9,
            optimizer: str = "SGD",
            weight_decay: float = 5e-4,
            lr_decay: float = 0.2,
            lr_milestones: list = None,
            prevent_resume: bool = False,
            transfer_from_path: str = "",
            freeze: bool = False,
            reset_linear: bool = False,
            comment: str = ""):
    device = 'cuda' if torch.cuda.is_available() and not force_cpu else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Model
    print('==> Building model..', flush=True)
    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    if transfer_from_path:
        comment += ".transfer"
        model, _, _ = load_model(model, transfer_from_path)
        if reset_linear:
            model.module.linear.reset_parameters()
        if freeze:
            model.module.freeze(exclude_linear=True)
    if not prevent_resume:
        path = "./checkpoint/ckpt.{}.pth".format(comment)
        if os.path.isfile(path):
            model, best_acc, start_epoch = load_model(model, path)

    criterion = nn.CrossEntropyLoss()
    if optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(),
                               lr=lr,
                               weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(),
                              lr=lr,
                              momentum=momentum,
                              weight_decay=weight_decay)
    if lr_milestones:
        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=lr_milestones,
                                                         gamma=lr_decay
                                                         )  # learning rate decay

    print('==> Starting model {}'.format(comment), flush=True)
    train_stats = []
    for epoch in range(start_epoch, max(num_epochs, start_epoch)):
        print("Is connected:", dj.conn().is_connected)
        train_acc, train_loss = train_loop(model, criterion, device, optimizer,
                                           data_loader=dataloaders["train"], epoch=epoch, add_noise=add_noise,
                                           noise_std=noise_std, noise_snr=noise_snr)
        dev_acc, dev_loss, best_acc = test_loop(model, criterion, device, data_loader=dataloaders["val"], epoch=epoch,
                                                comment=comment, best_acc=best_acc, add_noise=add_noise,
                                                noise_std=noise_std, noise_snr=noise_snr)
        if lr_milestones:
            train_scheduler.step(epoch=epoch)
        train_stats.append(dict(train_acc=train_acc, train_loss=train_loss, dev_acc=dev_acc, dev_loss=dev_loss))

    # test the final model
    dev_noise_acc, dev_noise_loss = test_model(model=model, path="./checkpoint/ckpt.{}.pth".format(comment),
                                               criterion=criterion, device=device, data_loader=dataloaders["val"],
                                               noise_test=noise_test)
    # test the final model
    test_acc, test_loss = test_model(model=model, path="./checkpoint/ckpt.{}.pth".format(comment),
                                     criterion=criterion, device=device, data_loader=dataloaders["test"])
    test_stat = dict(test_acc=test_acc, test_loss=test_loss, dev_noise_acc=dev_noise_acc, dev_noise_loss=dev_noise_loss)

    return test_acc, (train_stats, test_stat), {'net': model.state_dict(), 'acc': best_acc, 'epoch': epoch}
