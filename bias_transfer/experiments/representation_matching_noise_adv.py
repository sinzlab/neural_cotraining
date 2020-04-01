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
                # 1.0,
                # 10.0,
                5.0,
        ):
            matching_options = {"representation": "conv_rep",
                                "criterion": "cosine",
                                "second_noise_std": {(0, 1.0): 1.0},
                                "lambda": reg_factor}
            name = "Noise Augmented + Repr. Matching ({}) + Adv. Regression".format(reg_factor)
            experiments[
                Description(name=name, seed=seed)] = Experiment(
                dataset=dataset.DatasetConfig(description="", dataset_cls="CIFAR100"),
                model=model.ModelConfig(
                    description="Noise Adv Regression",
                    dataset_cls="CIFAR100",
                    noise_adv_regression=True),
                trainer=trainer.TrainerConfig(description=name,
                                              representation_matching=matching_options,
                                              noise_adv_classification=False,
                                              noise_adv_regression=True,
                                              noise_adv_loss_factor=1.0,
                                              noise_adv_gamma=10.0,
                                              **noise_type),
                seed=seed)

        transfer_experiments[Description(name=name + " -> Transfer (Reset)", seed=seed)] = TransferExperiment(
            [
                experiments[Description(name=name, seed=seed)],
                baseline.experiments[Description(name="Transfer + Reset", seed=seed)]])
