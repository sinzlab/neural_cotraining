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
from bias_transfer.experiments import Description
from bias_transfer.analysis.representation_analysis import RepresentationAnalyser
import argparse


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


def analyse(experiment_key):
    from bias_transfer.experiments.representation_analysis import experiments
    dataset_cls = "CIFAR10"
    if experiment_key == "clean":
        exp = experiments[Description(name=dataset_cls + ": Clean", seed=42)]
    if experiment_key == "noisy":
        exp = experiments[Description(name=dataset_cls + ": Noise Augmented", seed=42)]
    if experiment_key == "rep_matching":
        exp = experiments[Description(name=dataset_cls + ": Noise Augmented + Repr. Matching", seed=42)]
    if experiment_key == "adv_regression":
        exp = experiments[Description(name=dataset_cls + ": Noise Augmented + Noise Adv Regession", seed=42)]

    # val_analyser = RepresentationAnalyser(experiment=exp, table=TrainedModel(), dataset="val",
    #                                       plot_style="lightpaper")
    # clean_indices = val_analyser.corr_matrix(mode="clean")
    # for i in range(1, 21):
    #     noise_level = 0.05 * i
    #     val_analyser.corr_matrix(mode="noisy", noise_level=noise_level)
    #     val_analyser.corr_matrix(mode="noisy", noise_level=noise_level, sorted_indices=clean_indices)
    #
    # del val_analyser
    train_analyser = RepresentationAnalyser(experiment=exp, table=TrainedModel(), dataset="train",
                                            plot_style="lightpaper")
    pca_clean = train_analyser.dim_reduction(noise_level=0.0, method="pca", mode="clean")
    for i in range(1, 11):
        noise_level = 0.05 * i
        train_analyser = RepresentationAnalyser(experiment=exp, table=TrainedModel(), dataset="train",
                                                plot_style="lightpaper")
        # train_analyser.dim_reduction(noise_level=noise_level, method="pca", mode="noisy")
        train_analyser.dim_reduction(noise_level=noise_level, method="pca", mode="noisy", pca=pca_clean)
    # train_analyser.dim_reduction(noise_level=0.0, method="tsne", mode="clean")
    # for i in range(1, 11):
    #     noise_level = 0.05 * i
    #     train_analyser = RepresentationAnalyser(experiment=exp, table=TrainedModel(), dataset="train",
    #                                             plot_style="lightpaper")
    #     train_analyser.dim_reduction(noise_level=noise_level, method="tsne", mode="noisy")


def main():
    from bias_transfer.experiments.representation_matching import transfer_experiments as rep_configs
    from bias_transfer.experiments.representation_matching_noise_adv import transfer_experiments as rep_adv_configs
    from bias_transfer.experiments.representation_analysis import transfer_experiments as rep_analysis_configs
    from bias_transfer.experiments.noise_adv_training import transfer_experiments as adv_configs
    from bias_transfer.experiments.self_attention import transfer_experiments as attn_configs
    from bias_transfer.experiments.baseline import transfer_experiments as baseline_configs

    fill_tables(baseline_configs)
    fill_tables(rep_analysis_configs)
    fill_tables(rep_configs)
    fill_tables(rep_adv_configs)
    fill_tables(adv_configs)
    fill_tables(attn_configs)
    run_all_experiments(baseline_configs)
    run_all_experiments(rep_analysis_configs)
    run_all_experiments(rep_configs)
    run_all_experiments(rep_adv_configs)
    run_all_experiments(adv_configs)
    run_all_experiments(attn_configs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Running pre-defined experiments or analysis')
    parser.add_argument('--analysis', dest='analysis', action='store', default="", type=str,
                        help='name of experiment to analyse')

    args = parser.parse_args()
    if args.analysis:
        analyse(args.analysis)
    else:
        main()
