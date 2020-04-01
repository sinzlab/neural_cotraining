from . import Description
from bias_transfer.configs.transfer_experiment import TransferExperiment
from bias_transfer.configs.experiment import Experiment
from bias_transfer.configs import model, dataset, trainer
from . import baseline

experiments = {}
transfer_experiments = {}

for seed in (8,
             13,
             42):
    # Noise augmentation:
    for noise_type in (
            {"add_noise": True, "noise_snr": None,
             "noise_std": {0.08: 0.1, 0.12: 0.1, 0.18: 0.1, 0.26: 0.1, 0.38: 0.1, -1: 0.5}},
    ):
        for noise_adv in (
                {"noise_adv_classification": False, "noise_adv_regression": False},
                {"noise_adv_classification": True, "noise_adv_regression": False},
                {"noise_adv_classification": False, "noise_adv_regression": True},):
            for loss_factor in (1.0,
                                0.1,
                                10.0,
                                2.0,
                                0.5,
                                5.0
                                ):
                for gamma in (
                        10.0,
                        50.0,
                        1.0,
                        5.0
                ):
                    name = "Noise"
                    if noise_adv["noise_adv_classification"]:
                        name += " + Adv Classification"
                        m = model.ModelConfig(description="Noise Adv Classification",
                                              dataset_cls="CIFAR100",
                                              noise_adv_classification=True)
                    else:
                        name += " + Adv Regression"
                        m = model.ModelConfig(
                            description="Noise Adv Regression",
                            dataset_cls="CIFAR100",
                            noise_adv_regression=True)
                    name += " (lambda {} gamma {})".format(loss_factor, gamma)
                    experiments[
                        Description(name=name, seed=seed)] = Experiment(
                        dataset=dataset.DatasetConfig(description="", dataset_cls="CIFAR100"),
                        model=m,
                        trainer=trainer.TrainerConfig(description=name,
                                                      noise_adv_loss_factor=loss_factor,
                                                      noise_adv_gamma=gamma,
                                                      **noise_adv,
                                                      **noise_type),
                        seed=seed)

                    # The transfer experiments:
                    transfer_experiments[Description(name=name + "-> Reset", seed=seed)] = TransferExperiment(
                        [experiments[Description(name=name, seed=seed)],
                         baseline.experiments[Description(name="Transfer + Reset", seed=seed)]])
