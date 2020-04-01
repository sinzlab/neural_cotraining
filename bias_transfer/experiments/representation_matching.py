from . import Description
from bias_transfer.configs.transfer_experiment import TransferExperiment
from bias_transfer.configs.experiment import Experiment
from bias_transfer.configs import model, dataset, trainer
from . import baseline

experiments = {}
transfer_experiments = {}

for seed in (42,
             8,
             13
             ):
    # Noise augmentation:
    for noise_type in (
            {"add_noise": True, "noise_snr": None,
             "noise_std": {0.08: 0.1, 0.12: 0.1, 0.18: 0.1, 0.26: 0.1, 0.38: 0.1, -1: 0.5}},
    ):
        # Representation matching:
        for reg_factor in (
                0.5,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
        ):
            matching_options = {"representation": "conv_rep",
                                "criterion": "mse",
                                "second_noise_std": {(0, 0.5): 1.0},
                                "only_for_clean": True,
                                "lambda": reg_factor}
            name = "Noise Augmented + Repr. Matching (clean only; Lambda {}; noise (0,0.5); Euclidean distance)".format(reg_factor)
            title = "Noise Augmented + Repr. Matching (clean only; Lambda {}; noise (0,0.5); Euclidean distance)".format(reg_factor)
            experiments[
                Description(name=title, seed=seed)] = Experiment(
                dataset=dataset.DatasetConfig(description="", dataset_cls="CIFAR100"),
                model=model.ModelConfig(description="", dataset_cls="CIFAR100"),
                trainer=trainer.TrainerConfig(description=name,
                                              representation_matching=matching_options,
                                              **noise_type),
                seed=seed)

            # The transfer experiments:
            transfer_experiments[Description(name=title + " -> Transfer (Reset)", seed=seed)] = TransferExperiment(
                [
                    experiments[Description(name=title, seed=seed)],
                    baseline.experiments[Description(name="Transfer + Reset", seed=seed)]])
# # find lambda
# for seed in (42,
#              8,
#              13
#              ):
#     # Noise augmentation:
#     for noise_type in (
#             {"add_noise": True, "noise_snr": None,
#              "noise_std": {0.08: 0.1, 0.12: 0.1, 0.18: 0.1, 0.26: 0.1, 0.38: 0.1, -1: 0.5}},
#     ):
#         # Representation matching:
#         for reg_factor in (
#                 1.0,
#                 10.0,
#                 5.0,
#         ):
#             matching_options = {"representation": "conv_rep",
#                                 "criterion": "cosine",
#                                 "second_noise_std": {(0, 1.0): 1.0},
#                                 "lambda": reg_factor}
#             name = "Noise Augmented + Repr. Matching ({})".format(reg_factor)
#             title = "Noise Augmented + Repr. Matching (clean and noisy; Lambda {})".format(reg_factor)
#             experiments[
#                 Description(name=title, seed=seed)] = Experiment(
#                 dataset=dataset.DatasetConfig(description="", dataset_cls="CIFAR100"),
#                 model=model.ModelConfig(description="", dataset_cls="CIFAR100"),
#                 trainer=trainer.TrainerConfig(description=name,
#                                               representation_matching=matching_options,
#                                               **noise_type),
#                 seed=seed)
#
#             # The transfer experiments:
#             transfer_experiments[Description(name=title + " -> Transfer (Reset)", seed=seed)] = TransferExperiment(
#                 [
#                     experiments[Description(name=title, seed=seed)],
#                     baseline.experiments[Description(name="Transfer + Reset", seed=seed)]])
#
# # Apply Representation matching only to clean data:
# for seed in (42,
#              8,
#              13
#              ):
#     # Noise augmentation:
#     for noise_type in (
#             {"add_noise": True, "noise_snr": None,
#              "noise_std": {0.08: 0.1, 0.12: 0.1, 0.18: 0.1, 0.26: 0.1, 0.38: 0.1, -1: 0.5}},
#     ):
#         # Representation matching:
#         for reg_factor in (
#                 # 1.0,
#                 # 10.0,
#                 5.0,
#         ):
#             for max_noise in (0.5, 1.0, 1.5, 2.0):
#                 matching_options = {"representation": "conv_rep",
#                                     "criterion": "cosine",
#                                     "second_noise_std": {(0, max_noise): 1.0},
#                                     "only_for_clean": True,
#                                     "lambda": reg_factor}
#                 name = "Noise Augmented + Repr. Matching ({}; on clean only) ".format(reg_factor)
#                 title = "Noise Augmented + Repr. Matching (on clean only, noise range (0.0,{})) ".format(max_noise)
#                 experiments[
#                     Description(name=title, seed=seed)] = Experiment(
#                     dataset=dataset.DatasetConfig(description="", dataset_cls="CIFAR100"),
#                     model=model.ModelConfig(description="", dataset_cls="CIFAR100"),
#                     trainer=trainer.TrainerConfig(description=name,
#                                                   representation_matching=matching_options,
#                                                   **noise_type),
#                     seed=seed)
#
#                 # The transfer experiments:
#                 transfer_experiments[Description(name=title + " -> Transfer (Reset)", seed=seed)] = TransferExperiment(
#                     [
#                         experiments[Description(name=title, seed=seed)],
#                         baseline.experiments[Description(name="Transfer + Reset", seed=seed)]])
#
# # Apply Representation matching only to clean data:
# for seed in (42,
#              8,
#              13
#              ):
#     # Noise augmentation:
#     for noise_type in (
#             {"add_noise": True, "noise_snr": None,
#              "noise_std": {0.08: 0.1, 0.12: 0.1, 0.18: 0.1, 0.26: 0.1, 0.38: 0.1, -1: 0.5}},
#     ):
#         # Representation matching:
#         for reg_factor in (
#                 # 1.0,
#                 # 10.0,
#                 5.0,
#         ):
#             for criterion in ("cosine", "euclidean"):
#                 matching_options = {"representation": "conv_rep",
#                                     "criterion": criterion,
#                                     "second_noise_std": {(0, 1.0): 1.0},
#                                     "only_for_clean": True,
#                                     "lambda": reg_factor}
#                 name = "Noise Augmented + Repr. Matching ({}; on clean only{}) ".format(reg_factor,
#                                                                                         " MSE" if criterion != "cosine" else "")
#                 title = "Noise Augmented + Repr. Matching ({} distance) ".format(criterion)
#                 experiments[
#                     Description(name=title, seed=seed)] = Experiment(
#                     dataset=dataset.DatasetConfig(description="", dataset_cls="CIFAR100"),
#                     model=model.ModelConfig(description="", dataset_cls="CIFAR100"),
#                     trainer=trainer.TrainerConfig(description=name,
#                                                   representation_matching=matching_options,
#                                                   **noise_type),
#                     seed=seed)
#
#                 # The transfer experiments:
#                 transfer_experiments[Description(name=title + " -> Transfer (Reset)", seed=seed)] = TransferExperiment(
#                     [
#                         experiments[Description(name=title, seed=seed)],
#                         baseline.experiments[Description(name="Transfer + Reset", seed=seed)]])
#
# # No noise with labels but representation matching:
# for seed in (42,
#              8,
#              13
#              ):
#     # Noise augmentation:
#     for noise_type in (
#             {"add_noise": True, "noise_snr": None,
#              "noise_std": None},
#     ):
#         # Representation matching:
#         for reg_factor in (
#                 # 1.0,
#                 # 10.0,
#                 5.0,
#         ):
#             matching_options = {"representation": "conv_rep",
#                                 "criterion": "cosine",
#                                 "second_noise_std": {(0, 1.0): 1.0},
#                                 "only_for_clean": True,
#                                 "lambda": reg_factor}
#             name = "Clean + Repr. Matching ({})".format(reg_factor)
#             title = "Clean + Repr. Matching"
#             experiments[
#                 Description(name=title, seed=seed)] = Experiment(
#                 dataset=dataset.DatasetConfig(description="", dataset_cls="CIFAR100"),
#                 model=model.ModelConfig(description="", dataset_cls="CIFAR100"),
#                 trainer=trainer.TrainerConfig(description=name,
#                                               representation_matching=matching_options,
#                                               **noise_type),
#                 seed=seed)
#
#             # The transfer experiments:
#             transfer_experiments[Description(name=title + " -> Transfer (Reset)", seed=seed)] = TransferExperiment(
#                 [
#                     experiments[Description(name=title, seed=seed)],
#                     baseline.experiments[Description(name="Transfer + Reset", seed=seed)]])
