from . import Description
from bias_transfer.configs.transfer_experiment import TransferExperiment
from bias_transfer.configs.experiment import Experiment
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
