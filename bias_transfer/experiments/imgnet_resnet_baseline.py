from . import Description
from bias_transfer.configs.experiment import Experiment
from bias_transfer.configs import model, dataset, trainer

experiments = {}

for seed in (8,):
    # Clean baseline:
    experiments[Description(name="Clean", seed=seed)] = Experiment(
        dataset=dataset.TinyImageNet(description=""),
        model=model.TinyImageNet(description=""),
        trainer=trainer.TrainerConfig(description=""),
        seed=seed)