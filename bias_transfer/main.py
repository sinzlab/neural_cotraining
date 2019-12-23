"""
Initial setup based on https://github.com/kuangliu/pytorch-cifar
and https://github.com/weiaicunzai/pytorch-cifar100
"""

from bias_transfer.dataset import dataset_loader
from bias_transfer.config import *
from bias_transfer.trainer import trainer
from bias_transfer.models.resnet import resnet_builder

import os
import datajoint as dj

dj.config['database.host'] = os.environ['DJ_HOST']
dj.config['database.user'] = os.environ['DJ_USER']
dj.config['database.password'] = os.environ['DJ_PASS']
dj.config['enable_python_native_blobs'] = True
# dj.config['schema_name'] = "anix_nnfabrik_bias_transfer"
dj.config['schema_name'] = "anix_nnfabrik_bias_transfer_test"

import nnfabrik as nnf
# nnf.config['repos'] = ['/notebooks/nnfabrik', '/notebooks/ml-utils', '/notebooks/bias_transfer']
from nnfabrik.main import *
from bias_transfer.tables.transfer import TrainedTransferModel


def fill_tables(config):
    architect = dict(fabrikant_name="Arne Nix", email="arnenix@googlemail.com", affiliation="sinzlab", dj_username="anix")
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


if __name__ == "__main__":
    config = Config(optimizer="Adam", lr=0.0003, lr_decay=0.5, num_epochs=1, apply_data_augmentation=True,
                    apply_data_normalization=False, transfer=True, freeze=True, reset_linear=False)
    fill_tables(config)
    # run_manually()
    # TrainedModel().populate(display_progress=True, reserve_jobs=True, order="random")
    TrainedTransferModel().populate(display_progress=True, reserve_jobs=True, order="random")
