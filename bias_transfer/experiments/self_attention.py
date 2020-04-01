from . import Description
from bias_transfer.configs.transfer_experiment import TransferExperiment
from bias_transfer.configs.experiment import Experiment
from bias_transfer.configs import model, dataset, trainer

experiments = {}
transfer_experiments = {}

for seed in (42,):
    # Clean baseline:
    experiments[Description(name="Convolution", seed=seed)] = Experiment(
        dataset=dataset.DatasetConfig(description="", dataset_cls="CIFAR100"),
        model=model.ModelConfig(description="", dataset_cls="CIFAR100"),
        trainer=trainer.TrainerConfig(description=""),
        seed=seed)
    experiments[Description(name="Self-Attention", seed=seed)] = Experiment(
        dataset=dataset.DatasetConfig(description="", dataset_cls="CIFAR100"),
        model=model.ModelConfig(description="", dataset_cls="CIFAR100", self_attention=True),
        trainer=trainer.TrainerConfig(description=""),
        seed=seed)
    transfer_experiments[Description(name="Convolution", seed=seed)] = TransferExperiment(
        [experiments[Description(name="Convolution", seed=seed)]])
    transfer_experiments[Description(name="Self-Attention", seed=seed)] = TransferExperiment(
        [experiments[Description(name="Self-Attention", seed=seed)]])
