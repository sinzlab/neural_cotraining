from . import Description
from bias_transfer.bias_transfer.configs.transfer_experiment import TransferExperiment
from bias_transfer.bias_transfer.configs.experiment import Experiment
from bias_transfer.configs import model, dataset, trainer
from . import baseline

experiments = {}
transfer_experiments = {}

for seed in (42,
             # 8,
             # 13
             ):
    # Noise augmentation:
    for noise_type in (
            {"add_noise": True, "noise_snr": None,
             "noise_std": {0.08: 0.1, 0.12: 0.1, 0.18: 0.1, 0.26: 0.1, 0.38: 0.1, -1: 0.5}},
    ):
        # Representation matching:
        for reg_factor in (
                1.0,
                10.0,
                100.0,
        ):
            matching_options = {"representation": "conv_rep",
                                "criterion": "cosine",
                                "second_noise_std": {(0, 1.0): 1.0},
                                "lambda": reg_factor}
            name = "Noise Augmented + Repr. Matching ({})".format(reg_factor)
            experiments[
                Description(name=name, seed=seed)] = Experiment(
                dataset=dataset.CIFAR100(description="Default"),
                model=model.CIFAR100(description="Default"),
                trainer=trainer.TrainerConfig(description=name,
                                              representation_matching=matching_options,
                                              # freeze=("readout",),
                                              # reset_linear=False,
                                              **noise_type),
                seed=seed)

            # The transfer experiments:
            transfer_experiments[Description(name=name, seed=seed)] = TransferExperiment(
                [
                    baseline.experiments[Description(name="Noise Augmented", seed=seed)],
                    experiments[Description(name=name, seed=seed)],
                    baseline.experiments[Description(name="Transfer", seed=seed)]])
            transfer_experiments[Description(name=name + " -> Reset ", seed=seed)] = TransferExperiment(
                [
                    experiments[Description(name=name, seed=seed)],
                    baseline.experiments[Description(name="Transfer + Reset", seed=seed)]])
