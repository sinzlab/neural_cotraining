from . import Description
from bias_transfer.configs.config import Config
from bias_transfer.configs.experiment import Experiment
from bias_transfer.configs import model, dataset, trainer

configs = {}
experiments = {}

# Clean baseline:
configs[Description(name="Clean", seed=42)] = Config(
    dataset=dataset.CIFAR100(description="Default"),
    model=model.CIFAR100(description="Default"),
    trainer=trainer.TrainerConfig(description="Default", num_epochs=1,
                                  noise_test={
                                      "noise_snr": [{1.0: 1.0}],
                                      "noise_std": [{0.1: 1.0}]
                                  }),
    seed=42)
experiments[Description(name="Clean", seed=42)] = configs[Description(name="Clean", seed=42)]

# Transfer back to clean data:
configs[Description(name="Transfer", seed=42)] = Config(
    dataset=dataset.CIFAR100(description="Default"),
    model=model.CIFAR100(description="Default"),
    trainer=trainer.TrainerConfig(description="Transfer", freeze=("core",), num_epochs=1,
                                  noise_test={
                                      "noise_snr": [{1.0: 1.0}],
                                      "noise_std": [{0.1: 1.0}]
                                  }),
    seed=42)

experiments[Description(name="Clean -> Transfer", seed=42)] = Experiment(
    [configs[Description(name="Clean", seed=42)],
     configs[Description(name="Transfer", seed=42)]])

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
