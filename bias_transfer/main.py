"""
Initial setup based on https://github.com/kuangliu/pytorch-cifar
and https://github.com/weiaicunzai/pytorch-cifar100
"""
from datetime import datetime
import os
import copy
import datajoint as dj
from datajoint.errors import LostConnectionError

dj.config['database.host'] = os.environ['DJ_HOST']
dj.config['database.user'] = os.environ['DJ_USER']
dj.config['database.password'] = os.environ['DJ_PASS']
dj.config['enable_python_native_blobs'] = True
dj.config['schema_name'] = "anix_nnfabrik_bias_transfer1"
# dj.config['schema_name'] = "anix_nnfabrik_bias_transfer_adv_noise"
# dj.config['schema_name'] = "anix_nnfabrik_bias_transfer_test"

from bias_transfer.tables.trained_model import *
from bias_transfer.tables.trained_transfer_model import *


def fill_tables(configs: dict):
    architect = dict(fabrikant_name=os.environ['USER'],
                     email=os.environ['EMAIL'],
                     affiliation=os.environ['AFFILIATION'],
                     dj_username=os.environ['DJ_USER'])
    Fabrikant().insert1(architect, skip_duplicates=True)
    for config in configs.values():
        config.add_to_table()


def run_experiments(configs, train_table, order="random", level=0):
    restrictions = []
    for config in configs.values():
        restr = config.get_restrictions()
        if len(restr) > level:
            restrictions.append(restr[level])
    try:
        train_table.populate(restrictions, display_progress=True, reserve_jobs=True, order=order)
    except LostConnectionError:
        raise LostConnectionError("Connection to database lost at {}".format(datetime.now()))


def run_all_experiments(configs):
    run_experiments(configs, TrainedModel(), level=0)
    CollapsedTrainedModel().populate()
    run_experiments(configs, TrainedTransferModel(), level=1)
    CollapsedTrainedTransferModel().populate()
    run_experiments(configs, TrainedTransferModel2(), level=2)


def analyze(configs, train_table):
    restrictions = []
    for config in configs.values():
        restrictions.append(dj.AndList(config.get_key()))


if __name__ == "__main__":
    from bias_transfer.experiments.representation_matching import experiments as rep_configs
    # from bias_transfer.experiments.noise_adv_training import experiments as adv_configs
    from bias_transfer.experiments.self_attention import experiments as attn_configs

    fill_tables(rep_configs)
    # fill_tables(adv_configs)
    fill_tables(attn_configs)
    run_all_experiments(rep_configs)
    # run_all_experiments(adv_configs)
    run_all_experiments(attn_configs)
