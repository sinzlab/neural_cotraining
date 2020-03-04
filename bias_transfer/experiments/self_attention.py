from . import Experiment, TransferExperiment, Description
from bias_transfer.configs import model, dataset, trainer

experiments = {}
transfer_experiments = {}

for seed in (42,):
    # Clean baseline:
    experiments[Description(name="Convolution", seed=seed)] = Experiment(
        dataset=dataset.CIFAR100(description="Default", batch_size=64),
        model=model.CIFAR100(description="Default", type=26),
        trainer=trainer.TrainerConfig(description="Default"),
        seed=seed)
    experiments[Description(name="Self-Attention", seed=seed)] = Experiment(
        dataset=dataset.CIFAR100(description="Default", batch_size=64),
        model=model.CIFAR100(description="Self-Attention", self_attention=True, type=26),
        trainer=trainer.TrainerConfig(description="Default"),
        seed=seed)
    transfer_experiments[Description(name="Convolution", seed=seed)] = TransferExperiment(
        [experiments[Description(name="Convolution", seed=seed)]])
    transfer_experiments[Description(name="Self-Attention", seed=seed)] = TransferExperiment(
        [experiments[Description(name="Self-Attention", seed=seed)]])

    #
    # # Transfer back to clean data:
    # configs[Description(name="Transfer", seed=seed)] = Config(
    #     dataset=dataset.CIFAR100(description="Default"),
    #     model=model.CIFAR100(description="Default"),
    #     trainer=trainer.TrainerConfig(description="Transfer", freeze=("core",)),
    #     seed=seed)
    # configs[Description(name="Transfer + Reset", seed=seed)] = Config(
    #     dataset=model.CIFAR100(description="Default"),
    #     model=model.CIFAR100(description="Default"),
    #     trainer=trainer.TrainerConfig(description="Transfer + Reset", freeze=("core",), reset_linear=True),
    #     seed=seed)
    #
    # # Noise augmentation:
    # for noise_type in (
    #         # {"add_noise": True, "noise_snr": {1.0: 0.5, None: 0.5}, "noise_std": None},
    #         #            {"add_noise": True, "noise_snr": {1.0: 0.9, None: 0.1}, "noise_std": None},
    #         #            {"add_noise": True, "noise_snr": {1.0: 1.0}, "noise_std": None},
    #         #            {"add_noise": True, "noise_snr": None, "noise_std": {0.5: 0.5, None: 0.5}},
    #         # {"add_noise": True, "noise_snr": None,
    #         #  "noise_std": {0.08: 0.2, 0.12: 0.2, 0.18: 0.2, 0.26: 0.2, 0.38: 0.2}},
    #         {"add_noise": True, "noise_snr": None,
    #          "noise_std": {0.08: 0.1, 0.12: 0.1, 0.18: 0.1, 0.26: 0.1, 0.38: 0.1, -1: 0.5}},
    # ):
    #     configs[Description(name="Noise Augmented", seed=seed)] = Config(
    #         dataset=dataset.CIFAR100(description="Default"),
    #         model=model.CIFAR100(description="Default"),
    #         trainer=trainer.TrainerConfig(description="Noise Augmented", **noise_type),
    #         seed=seed)
    #
    #     # The transfer experiments:
    #     experiments[Description(name="Noise Augmented", seed=seed)] = Experiment(
    #         [configs[Description(name="Noise Augmented", seed=seed)],
    #          configs[Description(name="Transfer", seed=seed)]])
    #     experiments[Description(name="Noise Augmented -> Reset", seed=seed)] = Experiment(
    #         [configs[Description(name="Noise Augmented", seed=seed)],
    #          configs[Description(name="Transfer + Reset", seed=seed)]])
