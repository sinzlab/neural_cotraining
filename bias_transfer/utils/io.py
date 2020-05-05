import os
import torch


def set_model(state_dict, model, ignore_missing):
    if ignore_missing:
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    else:
        model.load_state_dict(state_dict)
    return model


def load_model(path, model, ignore_missing=False):
    print("==> Loading model..", flush=True)
    assert os.path.isfile(path), "Error: no model file found!"
    state_dict = torch.load(path)
    return set_model(state_dict, model, ignore_missing)


def load_checkpoint(path, model, optimizer=None, ignore_missing=False):
    assert os.path.isfile(path), "Error: no checkpoint file found!"
    checkpoint = torch.load(path)
    model = set_model(checkpoint["net"], model, ignore_missing)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]
    print("==> Loading checkpoint from epoch {}".format(start_epoch), flush=True)
    return model, best_acc, start_epoch


def save_checkpoint(model, optimizer, acc, epoch, path, name):
    print("==> Saving..", flush=True)
    state = {
        "net": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "acc": acc,
        "epoch": epoch,
    }
    if not os.path.isdir(path):
        os.mkdir(path)
    torch.save(state, os.path.join(path, name))
