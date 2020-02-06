import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
import nnfabrik as nnf

import datajoint as dj

import os
from tqdm import tqdm

from bias_transfer.utils import stringify
import matplotlib.pyplot as plt
import numpy as np

# from slacker import Slacker
# from torch.utils.tensorboard import SummaryWriter



class TrainingReport:
    def __init__(self, label, icon_emoji=None):
        self.label = label
        self.icon_emoji = icon_emoji if icon_emoji else ':clipboard:'
        self.connect()
    def connect(self, slack_api_token=None):
        slack_api_token = slack_api_token if slack_api_token else os.environ['SLACK_TOKEN']
        self.slack = Slacker(slack_api_token)
        if self.slack.api.test().successful:
            print(f"Connected to {self.slack.team.info().body['team']['name']} workspace.", flush=True)
        else:
            print('Try Again!')
    def send_text(self, text, channel='trainingreport'):
        self.slack.chat.post_message(channel=channel, text=text,
                                     username=self.label, icon_emoji=self.icon_emoji)
    def send_figure(self):
        raise NotImplementedError()


def load_checkpoint(path, model, optimizer=None, ignore_missing=False):
    print('==> Loading checkpoint..', flush=True)
    assert os.path.isfile(path), 'Error: no checkpoint file found!'
    checkpoint = torch.load(path)
    if ignore_missing:
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in checkpoint['net'].items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    else:
        model.load_state_dict(checkpoint['net'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    return model, best_acc, start_epoch


def save_model(model, optimizer, acc, epoch, path, name):
    print('==> Saving..', flush=True)
    state = {
        'net': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir(path):
        os.mkdir(path)
    torch.save(state, os.path.join(path, name))


def apply_noise(x, device, std: dict = None, snr: dict = None, rnd_gen=None):
    with torch.no_grad():
        noise_levels = std if std else snr
        assert sum(noise_levels.values())-1.0 < 0.00001, "Percentage for noise levels should sum to one!"
        indices = torch.randperm(x.shape[0])
        applied_std = torch.zeros([x.shape[0], 1], device=device)
        start = 0
        for level, percentage in noise_levels.items():  # TODO: is this efficient enough?
            end = start + int(percentage * x.shape[0])
            if level is not None:  # option to deactivate noise for a fraction of the data
                if std is None:  # are we doing snr or std?
                    signal = torch.mean(x[indices[start:end]] * x[indices[start:end]], dim=[1, 2, 3],
                                        keepdim=True)  # for each dimension except batch
                    std = signal / level
                else:
                    std = torch.tensor(level, device=device)
                applied_std[indices[start:end]] = std.squeeze().unsqueeze(-1)
                std = std.expand_as(x[start:end])
                x[indices[start:end]] += torch.normal(mean=0.0, std=std, generator=rnd_gen)
            start = end
        x = torch.clamp(x, max=1.0, min=0.0)
    return x, applied_std


def train_loop(model, criterion, device, optimizer, data_loader, epoch: int, noise_criterion=None,
               config: dict = {}):
    model.train()
    train_loss, train_noise_loss, correct, total = 0, 0, 0, 0
    previous_steps = epoch * len(data_loader)
    total_steps = config.get("num_epochs") * len(data_loader)
    if hasattr(tqdm, '_instances'):
        tqdm._instances.clear()
    with tqdm(data_loader, desc='Train Epoch {}'.format(epoch)) as t:
        for batch_idx, (inputs, targets) in enumerate(t):
            if config.get("reset_linear_frequency", {}).get("batch"):
                if batch_idx % config.get("reset_linear_frequency", {})["batch"] == 0:
                    model.module.linear_readout.reset_parameters()
            inputs, targets = inputs.to(device), targets.to(device)
            if config.get("add_noise"):
                inputs, applied_std = apply_noise(inputs, device, std=config.get("noise_std"),
                                                  snr=config.get("noise_snr"))
            optimizer.zero_grad()
            if noise_criterion:
                progress = float(batch_idx + previous_steps) / total_steps
                noise_adv_lambda = 2. / (1. + np.exp(-config.get("noise_adv_gamma") * progress)) - 1
                outputs = model(inputs, noise_lambda=noise_adv_lambda)
            else:
                outputs = model(inputs)
            loss = criterion(outputs[0], targets)
            train_loss += loss.item()
            if noise_criterion:
                if config.get("noise_adv_classification"):
                    applied_std = (applied_std > 0.0).type(torch.FloatTensor).to(device=device)
                noise_loss = noise_criterion(outputs[2], applied_std)
                train_noise_loss += noise_loss.item()
                loss += config.get("noise_adv_loss_factor") * noise_loss
            loss.backward()
            optimizer.step()

            _, predicted = outputs[0].max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc = 100. * correct / total
            t.set_postfix(acc=acc, loss=train_loss / total, noise_loss=train_noise_loss / total)
    return acc, train_loss / total, train_noise_loss / total


def test_loop(model, criterion, device, data_loader, epoch,
              noise_criterion=None, seed: int = 42, config: dict = {}):
    model.eval()
    test_loss = 0
    test_noise_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(data_loader, desc='Eval Epoch {}'.format(epoch)) as t:
            if config.get("add_noise"):
                rnd_gen = torch.Generator(device=device)
                rnd_gen = rnd_gen.manual_seed(seed)  # so that we always have the same noise for evaluation!
            for batch_idx, (inputs, targets) in enumerate(t):
                inputs, targets = inputs.to(device), targets.to(device)
                if config.get("add_noise"):
                    inputs, applied_std = apply_noise(inputs, device, std=config.get("noise_std"),
                                                      snr=config.get("noise_snr"), rnd_gen=rnd_gen)
                outputs = model(inputs, compute_corr=config.get("compute_corr"))
                if config.get("compute_corr"):
                    return outputs[1]
                loss = criterion(outputs[0], targets)
                if noise_criterion:
                    if config.get("noise_adv_classification"):
                        applied_std = (applied_std > 0.0).type(torch.FloatTensor).to(device=device)
                    noise_loss = noise_criterion(outputs[2], applied_std)
                    test_noise_loss += noise_loss.item()

                test_loss += loss.item()
                _, predicted = outputs[0].max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                t.set_postfix(acc=100. * correct / total, loss=loss.item(),
                              noise_loss=noise_loss.item() if noise_criterion else None)

    # Save checkpoint.
    acc = 100. * correct / total
    return acc, test_loss, test_noise_loss


def test_model(model, path, criterion, device, data_loader, config: dict = {}, noise_test: bool = True):
    model, _, epoch = load_checkpoint(path, model)
    if config.get("noise_test") and noise_test:
        test_acc = {}
        test_loss = {}
        for n_type, n_vals in config.get("noise_test").items():
            test_acc[n_type] = {}
            test_loss[n_type] = {}
            for val in n_vals:
                val_str = stringify(val)
                test_acc[n_type][val_str], test_loss[n_type][val_str], _ = test_loop(model,
                                                                                        criterion,
                                                                                        device,
                                                                                        data_loader=data_loader,
                                                                                        epoch=epoch,
                                                                                        config={
                                                                                            "comment": "Final Eval",
                                                                                            "add_noise": True,
                                                                                            n_type: val})
    else:
        test_acc, test_loss, _ = test_loop(model, criterion, device,
                                              data_loader=data_loader, epoch=epoch,
                                              config={"comment": "Final Eval",
                                                      "add_noise": False})
    return test_acc, test_loss


def trainer(model, dataloaders, seed, uid="default", cb=None, **config):
    uid = nnf.utility.dj_helpers.make_hash(uid)
    device = 'cuda' if torch.cuda.is_available() and not config.get("force_cpu") else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = -1  # start from epoch 0 or last checkpoint epoch
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Model
    print('==> Building model..', flush=True)
    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)

    if config.get("optimizer") == "Adam":
        optimizer = optim.Adam(model.parameters(),
                               lr=config.get("lr"),
                               weight_decay=config.get("weight_decay"))
    else:
        optimizer = optim.SGD(model.parameters(),
                              lr=config.get("lr"),
                              momentum=config.get("momentum"),
                              weight_decay=config.get("weight_decay"))
    if config.get("lr_milestones"):
        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=config.get("lr_milestones"),
                                                         gamma=config.get("lr_decay")
                                                         )  # learning rate decay

    comment = config.get("comment", "")
    if config.get("transfer_from_path"):
        comment += ".transfer"
        model, _, _ = load_checkpoint(config.get("transfer_from_path"), model, ignore_missing=True)
        if config.get("reset_linear"):
            model.module.linear_readout.reset_parameters()
        if config.get("freeze"):
            model.module.freeze(config.get("freeze"))
    else:
        path = "./checkpoint/ckpt.{}.pth".format(uid)
        if os.path.isfile(path):
            model, best_acc, start_epoch = load_checkpoint(path, model, optimizer)

    criterion = nn.CrossEntropyLoss()
    if config.get("noise_adv_regression"):
        noise_criterion = nn.MSELoss()
    elif config.get("noise_adv_classification"):
        noise_criterion = nn.BCELoss()
    else:
        noise_criterion = None

    print('==> Starting model {}'.format(comment), flush=True)
    # if config.get("use_tensorboard"):
    #     # default `log_dir` is "runs" - we'll be more specific here
    #     writer = SummaryWriter('gs://anix-tensorboard-logs/logs/{}'.format(uid))
    train_stats = []
    for epoch in range(start_epoch + 1, config.get("num_epochs")):
        if config.get("reset_linear_frequency",{}).get("epoch"):
            if epoch % config.get("reset_linear_frequency", {})["epoch"] == 0:
                model.module.linear_readout.reset_parameters()
                print('Resetting readout', flush=True)
        if cb:
            cb()
        train_acc, train_loss, train_noise_loss = train_loop(model, criterion, device, optimizer,
                                                             data_loader=dataloaders["train"],
                                                             epoch=epoch,
                                                             noise_criterion=noise_criterion,
                                                             config=config)
        dev_acc, dev_loss, dev_noise_loss = test_loop(model, criterion, device,
                                                                data_loader=dataloaders["val"],
                                                                epoch=epoch,
                                                                noise_criterion=noise_criterion,
                                                                config=config)



        if dev_acc > best_acc:
            save_model(model, optimizer, dev_acc, epoch, "./checkpoint", "ckpt.{}.pth".format(uid))
            best_acc = dev_acc
        if config.get("lr_milestones"):
            train_scheduler.step(epoch=epoch)
        train_stats.append(dict(train_acc=train_acc, train_loss=train_loss, dev_acc=dev_acc, dev_loss=dev_loss))
        # if config.get("use_tensorboard"):
        #     writer.add_scalars("training_progress", train_stats[-1], epoch)


    # test the final model
    dev_noise_acc, dev_noise_loss = test_model(model=model, path="./checkpoint/ckpt.{}.pth".format(uid),
                                               criterion=criterion, device=device, data_loader=dataloaders["val"],
                                               config=config)
    # test the final model
    test_acc, test_loss = test_model(model=model, path="./checkpoint/ckpt.{}.pth".format(uid),
                                     criterion=criterion, device=device, data_loader=dataloaders["test"],
                                     config=config, noise_test=False)
    test_stat = dict(test_acc=test_acc, test_loss=test_loss, dev_noise_acc=dev_noise_acc, dev_noise_loss=dev_noise_loss)

    return test_acc, (train_stats, test_stat), {'net': model.state_dict(), 'acc': best_acc, 'epoch': epoch}
