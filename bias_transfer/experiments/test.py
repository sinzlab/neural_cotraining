from . import Description
from bias_transfer.configs.transfer_experiment import TransferExperiment
from bias_transfer.configs.experiment import Experiment
from bias_transfer.configs import model, dataset, trainer

experiments = {}
transfer_experiments = {}

# Clean baseline:
experiments[Description(name="Clean", seed=42)] = Experiment(
    dataset=dataset.DatasetConfig(description="", dataset_cls="CIFAR100"),
    model=model.ModelConfig(description="", dataset_cls="CIFAR100"),
    trainer=trainer.TrainerConfig(description="Default", num_epochs=1,
                                  noise_test={
                                      "noise_snr": [{1.0: 1.0}],
                                      "noise_std": [{0.1: 1.0}]
                                  }),
    seed=42)
transfer_experiments[Description(name="Clean", seed=42)] = experiments[Description(name="Clean", seed=42)]

# Transfer back to clean data:
experiments[Description(name="Transfer", seed=42)] = Experiment(
    dataset=dataset.DatasetConfig(description="", dataset_cls="CIFAR100"),
    model=model.ModelConfig(description="", dataset_cls="CIFAR100"),
    trainer=trainer.TrainerConfig(description="Transfer", freeze=("core",), num_epochs=1,
                                  noise_test={
                                      "noise_snr": [{1.0: 1.0}],
                                      "noise_std": [{0.1: 1.0}]
                                  }),
    seed=42)

transfer_experiments[Description(name="Clean -> Transfer", seed=42)] = TransferExperiment(
    [experiments[Description(name="Clean", seed=42)],
     experiments[Description(name="Transfer", seed=42)]])

