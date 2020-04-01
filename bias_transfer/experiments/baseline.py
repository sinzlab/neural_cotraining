from . import Description
from bias_transfer.configs.transfer_experiment import TransferExperiment
from bias_transfer.configs.experiment import Experiment
from bias_transfer.configs import model, dataset, trainer

experiments = {}
transfer_experiments = {}

for seed in (8, 13, 42):
    # Clean baseline:
    experiments[Description(name="Clean", seed=seed)] = Experiment(
        dataset=dataset.DatasetConfig(description="", dataset_cls="CIFAR100"),
        model=model.ModelConfig(description="", dataset_cls="CIFAR100"),
        trainer=trainer.TrainerConfig(description=""),
        seed=seed)
    transfer_experiments[Description(name="Clean", seed=seed)] = TransferExperiment(
        [experiments[Description(name="Clean", seed=seed)]])

    # Transfer back to clean data:
    experiments[Description(name="Transfer", seed=seed)] = Experiment(
        dataset=dataset.DatasetConfig(description="", dataset_cls="CIFAR100"),
        model=model.ModelConfig(description="", dataset_cls="CIFAR100"),
        trainer=trainer.TrainerConfig(description="Transfer", freeze=("core",)),
        seed=seed)
    experiments[Description(name="Transfer + Reset", seed=seed)] = Experiment(
        dataset=dataset.DatasetConfig(description="", dataset_cls="CIFAR100"),
        model=model.ModelConfig(description="", dataset_cls="CIFAR100"),
        trainer=trainer.TrainerConfig(description="Transfer + Reset", freeze=("core",), reset_linear=True),
        seed=seed)

    # Noise augmentation:
    for noise_type in (
            # {"add_noise": True, "noise_snr": {1.0: 0.5, None: 0.5}, "noise_std": None},
            #            {"add_noise": True, "noise_snr": {1.0: 0.9, None: 0.1}, "noise_std": None},
            #            {"add_noise": True, "noise_snr": {1.0: 1.0}, "noise_std": None},
            #            {"add_noise": True, "noise_snr": None, "noise_std": {0.5: 0.5, None: 0.5}},
            # {"add_noise": True, "noise_snr": None,
            #  "noise_std": {0.08: 0.2, 0.12: 0.2, 0.18: 0.2, 0.26: 0.2, 0.38: 0.2}},
            {"add_noise": True, "noise_snr": None,
             "noise_std": {0.08: 0.1, 0.12: 0.1, 0.18: 0.1, 0.26: 0.1, 0.38: 0.1, -1: 0.5}},
    ):
        experiments[Description(name="Noise Augmented", seed=seed)] = Experiment(
            dataset=dataset.DatasetConfig(description="", dataset_cls="CIFAR100"),
            model=model.ModelConfig(description="", dataset_cls="CIFAR100"),
            trainer=trainer.TrainerConfig(description="Noise Augmented", **noise_type),
            seed=seed)

        # The transfer experiments:
        # transfer_experiments[Description(name="Noise Augmented -> Transfer", seed=seed)] = TransferExperiment(
        #     [experiments[Description(name="Noise Augmented", seed=seed)],
        #      experiments[Description(name="Transfer", seed=seed)]])
        transfer_experiments[Description(name="Noise Augmented -> Transfer (Reset)", seed=seed)] = TransferExperiment(
            [experiments[Description(name="Noise Augmented", seed=seed)],
             experiments[Description(name="Transfer + Reset", seed=seed)]])
