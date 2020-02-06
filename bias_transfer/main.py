"""
Initial setup based on https://github.com/kuangliu/pytorch-cifar
and https://github.com/weiaicunzai/pytorch-cifar100
"""

import os
import copy
import datajoint as dj

dj.config['database.host'] = os.environ['DJ_HOST']
dj.config['database.user'] = os.environ['DJ_USER']
dj.config['database.password'] = os.environ['DJ_PASS']
dj.config['enable_python_native_blobs'] = True
# dj.config['schema_name'] = "anix_nnfabrik_bias_transfer"
dj.config['schema_name'] = "anix_nnfabrik_bias_transfer_adv_noise"
# dj.config['schema_name'] = "anix_nnfabrik_bias_transfer_test"

from bias_transfer.dataset import dataset_loader
from bias_transfer.config import *
from bias_transfer.trainer import trainer
from bias_transfer.models import resnet_builder

import nnfabrik as nnf
# nnf.config['repos'] = ['/notebooks/nnfabrik']
from nnfabrik.main import *
from bias_transfer.tables.base import *


def fill_tables(config):
    architect = dict(fabrikant_name="Arne Nix", email="arnenix@googlemail.com", affiliation="sinzlab",
                     dj_username="anix")
    Fabrikant().insert1(architect)
    Seed().insert1(dict(seed=42))
    Dataset().add_entry(dataset_fn="bias_transfer.dataset.dataset_loader",
                        dataset_config=config.data,
                        dataset_fabrikant="Arne Nix",
                        dataset_comment=config.data_comment[:256])
    Model().add_entry(model_fn="bias_transfer.models.resnet.resnet_builder",
                      model_config=config.model,
                      model_fabrikant="Arne Nix",
                      model_comment=config.model_comment[:256])
    Trainer().add_entry(trainer_fn="bias_transfer.trainer.trainer",
                        trainer_config=config.trainer,
                        trainer_fabrikant="Arne Nix",
                        trainer_comment=config.name[:256])
    # TrainedModel().populate(display_progress=True)


def run_manually():
    data_loaders, model, train_fct = nnf.builder.get_all_parts(dataset_fn=dataset_loader,
                                                               dataset_config=config.data,
                                                               trainer_fn=trainer,
                                                               trainer_config=config.trainer,
                                                               model_fn=resnet_builder,
                                                               model_config=config.model
                                                               )
    return train_fct(model, 42, train=data_loaders["train"], val=data_loaders["val"],
                     test=data_loaders["test"])


def get_matched_trainer_model(keys, restriction=None):
    matched_hashes = []
    trainers = Trainer().fetch("trainer_hash", "trainer_config", as_dict=True)
    models = Model().fetch("model_hash", "model_config", as_dict=True)
    for t in trainers:
        for m in models:
            match = True
            for k in keys:
                if t['trainer_config'].get(k) != m['model_config'].get(k):
                    match = False
                    break
                if restriction is not None:
                    if t['trainer_config'].get(k) != restriction:
                        match = False
                        break
            if match:
                matched_hashes.append({'trainer_hash': t['trainer_hash'], 'model_hash': m['model_hash']})
    return matched_hashes


def fill_transfer_experiments():
    fabrikant = dict(fabrikant_name="Arne Nix", email="arnenix@googlemail.com", affiliation="sinzlab",
                     dj_username="anix")
    Fabrikant().insert1(fabrikant, skip_duplicates=True)
    base_config = dict(optimizer="Adam",
                       lr=0.0003,
                       lr_decay=0.8,
                       num_epochs=200,
                       add_noise=False,
                       noise_snr=None,
                       noise_std=None,
                       apply_data_augmentation=True,
                       apply_data_normalization=False,
                       noise_adv_classification=False,
                       noise_adv_regression=False)
    # for seed in (42,):
    for seed in (42, 8, 13):
        base_config = copy.deepcopy(base_config)
        base_config["seed"] = seed
        for noise_type in ({"add_noise": False, "noise_snr": None, "noise_std": None},
                           {"add_noise": True, "noise_snr": {1.0: 0.5, None: 0.5}, "noise_std": None},
                           {"add_noise": True, "noise_snr": {1.0: 0.9, None: 0.1}, "noise_std": None},
                           {"add_noise": True, "noise_snr": {1.0: 1.0}, "noise_std": None},
                           {"add_noise": True, "noise_snr": None, "noise_std": {0.5: 0.5, None: 0.5}},
                           {"add_noise": True, "noise_snr": None,
                            "noise_std": {0.08: 0.2, 0.12: 0.2, 0.18: 0.2, 0.26: 0.2, 0.38: 0.2}},
                           {"add_noise": True, "noise_snr": None,
                            "noise_std": {0.08: 0.1, 0.12: 0.1, 0.18: 0.1, 0.26: 0.1, 0.38: 0.1, None: 0.5}},
                           ):
            for noise_adv in ({"noise_adv_classification": False, "noise_adv_regression": False},
                              # {"noise_adv_classification": True, "noise_adv_regression": False},
                              {"noise_adv_classification": False, "noise_adv_regression": True}):
                if not noise_type["add_noise"]:
                    if noise_adv["noise_adv_classification"]:
                        continue
                    if noise_adv["noise_adv_regression"]:
                        continue
                # one config to create the "robust" core
                config = copy.deepcopy(base_config)
                config.update(noise_type)
                config.update(noise_adv)
                # one config for transfer to clean data
                transfer_config = copy.deepcopy(base_config)
                transfer_config["transfer"] = True
                transfer_config["freeze"] = True
                transfer_config["reset_linear"] = False
                ConfigToTrainAndTransfer().add_entry(
                    [Config(**config), Config(**transfer_config)])  # Train all variants directly
                ConfigToTrainAndTransfer2().add_entry([Config(**base_config), Config(**config),
                                                       Config(**transfer_config)])  # Train all variants after directly


def fill_frozen_readout_experiment():
    fabrikant = dict(fabrikant_name="Arne Nix", email="arnenix@googlemail.com", affiliation="sinzlab",
                     dj_username="anix")
    Fabrikant().insert1(fabrikant, skip_duplicates=True)
    base_config = dict(optimizer="Adam",
                       lr=0.0003,
                       lr_decay=0.8,
                       num_epochs=200,
                       add_noise=False,
                       noise_snr=None,
                       noise_std=None,
                       apply_data_augmentation=True,
                       apply_data_normalization=False,
                       noise_adv_classification=False,
                       noise_adv_regression=False)
    for reset_linear in (False, True):
        for seed in (42, 8, 13):
            base_config = copy.deepcopy(base_config)
            base_config["seed"] = seed
            for noise_type in (  # {"add_noise": False, "noise_snr": None, "noise_std": None},
                    # {"add_noise": True, "noise_snr": {1.0: 0.5, None: 0.5}, "noise_std": None},
                    # {"add_noise": True, "noise_snr": {1.0: 0.9, None: 0.1}, "noise_std": None},
                    # {"add_noise": True, "noise_snr": {1.0: 1.0}, "noise_std": None},
                    # {"add_noise": True, "noise_snr": None, "noise_std": {0.5: 0.5, None: 0.5}},
                    {"add_noise": True, "noise_snr": None,
                     "noise_std": {0.08: 0.2, 0.12: 0.2, 0.18: 0.2, 0.26: 0.2, 0.38: 0.2}},
                    {"add_noise": True, "noise_snr": None,
                     "noise_std": {0.08: 0.1, 0.12: 0.1, 0.18: 0.1, 0.26: 0.1, 0.38: 0.1, None: 0.5}},
            ):
                for noise_adv in ({"noise_adv_classification": False, "noise_adv_regression": False},
                                  {"noise_adv_classification": True, "noise_adv_regression": False},
                                  {"noise_adv_classification": False, "noise_adv_regression": True}
                                  ):
                    if not noise_type["add_noise"]:
                        if noise_adv["noise_adv_classification"]:
                            continue
                        if noise_adv["noise_adv_regression"]:
                            continue
                    # one config to create the readout
                    config = copy.deepcopy(base_config)
                    # one config for transfer to noisy data (to train the robust core)
                    transfer_config = copy.deepcopy(base_config)
                    transfer_config["transfer"] = True
                    transfer_config["freeze"] = ("readout",)
                    transfer_config["reset_linear"] = False
                    transfer_config.update(noise_type)
                    transfer_config.update(noise_adv)
                    transfer2_config = copy.deepcopy(base_config)
                    transfer2_config["transfer"] = True
                    transfer2_config["freeze"] = ("core",)
                    transfer2_config["reset_linear"] = reset_linear
                    ConfigToTrainAndTransfer().add_entry(
                        [Config(**config), Config(**transfer_config)])  # Train all variants directly
                    ConfigToTrainAndTransfer2().add_entry([Config(**config), Config(**transfer_config),
                                                           Config(
                                                               **transfer2_config)])  # Train all variants after directly


def fill_random_readout_experiment():
    fabrikant = dict(fabrikant_name="Arne Nix", email="arnenix@googlemail.com", affiliation="sinzlab",
                     dj_username="anix")
    Fabrikant().insert1(fabrikant, skip_duplicates=True)
    base_config = dict(optimizer="Adam",
                       lr=0.0003,
                       lr_decay=0.8,
                       num_epochs=200,
                       add_noise=False,
                       noise_snr=None,
                       noise_std=None,
                       apply_data_augmentation=True,
                       apply_data_normalization=False,
                       noise_adv_classification=False,
                       noise_adv_regression=False)
    for seed in (42,):
    # for seed in (42, 8, 13):
        for reset_freq in ({},
                           {"epoch": 1},
                           {"epoch": 4},
                           {"batch": 8},
                           {"batch": 64}
        ):
            base_config = copy.deepcopy(base_config)
            base_config["seed"] = seed
            for noise_type in (  # {"add_noise": False, "noise_snr": None, "noise_std": None},
                    # {"add_noise": True, "noise_snr": {1.0: 0.5, None: 0.5}, "noise_std": None},
                    # {"add_noise": True, "noise_snr": {1.0: 0.9, None: 0.1}, "noise_std": None},
                    # {"add_noise": True, "noise_snr": {1.0: 1.0}, "noise_std": None},
                    # {"add_noise": True, "noise_snr": None, "noise_std": {0.5: 0.5, None: 0.5}},
                    {"add_noise": True, "noise_snr": None,
                     "noise_std": {0.08: 0.2, 0.12: 0.2, 0.18: 0.2, 0.26: 0.2, 0.38: 0.2}},
                    {"add_noise": True, "noise_snr": None,
                    "noise_std": {0.08: 0.1, 0.12: 0.1, 0.18: 0.1, 0.26: 0.1, 0.38: 0.1, None: 0.5}},
            ):
                for noise_adv in ({"noise_adv_classification": False, "noise_adv_regression": False},
                        # {"noise_adv_classification": True, "noise_adv_regression": False},
                        # {"noise_adv_classification": False, "noise_adv_regression": True}
                                  ):
                    if not noise_type["add_noise"]:
                        if noise_adv["noise_adv_classification"]:
                            continue
                        if noise_adv["noise_adv_regression"]:
                            continue
                    # one config to create the readout
                    config = copy.deepcopy(base_config)
                    config["freeze"] = ("readout",)
                    config["reset_linear"] = True
                    config["reset_linear_frequency"] = reset_freq
                    config.update(noise_type)
                    config.update(noise_adv)
                    # one config for transfer to noisy data (to train the robust core)
                    transfer_config = copy.deepcopy(base_config)
                    transfer_config["transfer"] = True
                    transfer_config["freeze"] = ("readout",)
                    transfer_config["reset_linear"] = True
                    transfer_config["reset_linear_frequency"] = reset_freq
                    transfer_config.update(noise_type)
                    transfer_config.update(noise_adv)
                    transfer2_config = copy.deepcopy(base_config)
                    transfer2_config["transfer"] = True
                    transfer2_config["freeze"] = ("core",)
                    transfer2_config["reset_linear"] = True
                    ConfigToTrainAndTransfer().add_entry(
                        [Config(**config), Config(**transfer2_config)])  # Train all variants directly
                    ConfigToTrainAndTransfer2().add_entry([Config(**base_config), Config(**transfer_config),
                                                           Config(
                                                               **transfer2_config)])  # Train all variants after directly


def run_transfer_experiments():
    # fill_transfer_experiments()
    TrainedModel().populate(display_progress=True, reserve_jobs=True, order="random")
    TrainedTransferModel().populate(display_progress=True, reserve_jobs=True, order="random")
    TrainedTransferModel2().populate(display_progress=True, reserve_jobs=True, order="random")


if __name__ == "__main__":
    run_transfer_experiments()
    # matched = get_matched_trainer_model(['noise_adv_classification', 'noise_adv_regression'])
    # TrainedModel().populate(matched, display_progress=True, reserve_jobs=True, order="random")
    #
    # transfer_matched = get_matched_trainer_model(['noise_adv_classification', 'noise_adv_regression'],
    #                                              restriction=False)  # Use the ones that were pretrained without adv_noise
    # for match in matched:
    #     for t in transfer_matched:
    #         for k, v in match.items():
    #             t["transfer_" + k] = v
    # TrainedTransferModel().populate(transfer_matched, display_progress=True, reserve_jobs=True, order="random")

    # without_transfer = (Trainer() & "NOT trainer_comment LIKE '%transfer%'")
    # hashes_wo_transfer = without_transfer.fetch("trainer_hash")
    # TrainedModel().populate(["trainer_hash='{}'".format(h) for h in hashes_wo_transfer],
    #                         display_progress=True, reserve_jobs=True, order="random")
    #
    # with_transfer = (Trainer() & "trainer_comment LIKE '%transfer%'")
    # hashes_w_transfer = with_transfer.fetch("trainer_hash")
    # TrainedTransferModel().populate(["transfer_trainer_hash='{}'".format(h) for h in hashes_w_transfer],
    #                                 display_progress=True, reserve_jobs=True, order="random")
