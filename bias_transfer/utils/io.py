import os
import torch


def set_model(pretrained_dict, model, ignore_missing, ignore_dim_mismatch=True):
    if ignore_missing:
        model_dict = model.state_dict()
        # 0. Try to match names by adding or removing prefix:
        first_key_pretrained = list(pretrained_dict.keys())[0].split(".")
        first_key_model = list(model_dict.keys())[0].split(".")
        remove_pretrained, add_pretrained = False, ""
        if first_key_pretrained[1:] == first_key_model:
            # prefix in pretrained
            remove_pretrained = True
        elif first_key_pretrained == first_key_model[1:]:
            # prefix in model
            add_pretrained = first_key_model[0]
        elif (
            first_key_pretrained[0] != first_key_model[0]
            and first_key_pretrained[1:] == first_key_model[1:]
        ):
            # prefix in both
            remove_pretrained = True
            add_pretrained = first_key_model[0]
        if remove_pretrained:
            state_dict_ = {}
            for k, v in pretrained_dict.items():
                state_dict_[".".join(k.split(".")[1:])] = v
            pretrained_dict = state_dict_
        if add_pretrained:
            state_dict_ = {}
            for k, v in pretrained_dict.items():
                state_dict_[".".join([add_pretrained] + k.split("."))] = v
            pretrained_dict = state_dict_
        # 1. filter out missing keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        left_out = set(model_dict.keys()) - set(pretrained_dict.keys())
        if left_out:
            print("Ignored missing keys:")
            for k in left_out:
                print(k)
        # 2. overwrite entries in the existing state dict
        for k, v in model_dict.items():
            if v.shape != pretrained_dict[k].shape and ignore_dim_mismatch:
                print("Ignored shape-mismatched parameters:", k)
                continue
            model_dict[k] = pretrained_dict[k]
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    else:
        model.load_state_dict(pretrained_dict)
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
