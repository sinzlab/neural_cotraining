from .base import BaseConfig
from nnfabrik.main import *


class TrainerConfig(BaseConfig):
    config_name = "trainer"
    table = Trainer()
    fn = "bias_transfer.trainer.trainer"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.force_cpu = kwargs.pop("force_cpu", False)
        self.use_tensorboard = kwargs.pop("use_tensorboard", False)
        self.optimizer = kwargs.pop("optimizer", "Adam")
        self.optimizer_options = kwargs.pop(
            "optimizer_options", {"amsgrad": False, "lr": 0.0003, "weight_decay": 5e-4}
        )
        self.lr_decay = kwargs.pop("lr_decay", 0.8)
        self.epoch = kwargs.pop("epoch", 0)
        self.scheduler = kwargs.pop("scheduler", None)
        self.scheduler_options = kwargs.pop("scheduler_options", {})
        self.patience = kwargs.pop("patience", 10)
        self.threshold = kwargs.pop("threshold", 0.0001)
        self.verbose = kwargs.pop("verbose", False)
        self.min_lr = kwargs.pop("min_lr", 0.0001)  # lr scheduler min learning rate
        self.threshold_mode = kwargs.pop("threshold_mode", "rel")
        self.train_cycler = kwargs.pop("train_cycler", "LongCycler")
        self.train_cycler_args = kwargs.pop("train_cycler_args", {})
        self.loss_functions = kwargs.pop(
            "loss_functions", {"img_classification": "CrossEntropyLoss"}
        )
        self.loss_sum = kwargs.pop("loss_sum", False)
        self.loss_weighing = kwargs.pop("loss_weighing", False)
        if (
            len(self.loss_functions) > 1
            or "img_classification" not in self.loss_functions.keys()
        ):
            self.threshold_mode = "abs"
            self.scale_loss = kwargs.pop("scale_loss", True)
            self.avg_loss = kwargs.pop("avg_loss", False)

        self.maximize = kwargs.pop(
            "maximize", True
        )  # if stop_function maximized or minimized

        self.mtl = kwargs.pop(
            "mtl", False
        )

        self.to_monitor = kwargs.pop(
            "to_monitor", ["img_classification"]
        )

        self.interval = kwargs.pop(
            "interval", 1
        )  # interval at which objective evaluated for early stopping
        self.max_iter = kwargs.pop(
            "max_iter", 100
        )  # maximum number of iterations (epochs)

        self.restore_best = kwargs.pop(
            "restore_best", True
        )  # in case of loading best model at the end of training
        self.lr_decay_steps = kwargs.pop(
            "lr_decay_steps", 3
        )  # Number of times the learning rate should be reduced before stopping the training.

        self.add_final_train_eval = kwargs.pop(
            "add_final_train_eval", True
        )
        self.add_final_val_eval = kwargs.pop(
            "add_final_val_eval", True
        )
        self.add_final_test_eval = kwargs.pop(
            "add_final_test_eval", True
        )
        self.track_training = kwargs.pop("track_training", False)
        # noise
        self.add_noise = kwargs.pop("add_noise", False)
        self.noise_std = kwargs.pop("noise_std", None)
        self.noise_snr = kwargs.pop("noise_snr", None)
        self.noise_test = kwargs.pop(
            "noise_test",
            {
                "noise_snr": [
                    {5.0: 1.0},
                    {4.0: 1.0},
                    {3.0: 1.0},
                    {2.0: 1.0},
                    {1.0: 1.0},
                    {0.5: 1.0},
                    {0.0: 1.0},
                ],
                "noise_std": [
                    {0.0: 1.0},
                    {0.05: 1.0},
                    {0.1: 1.0},
                    {0.2: 1.0},
                    {0.3: 1.0},
                    {0.5: 1.0},
                    {1.0: 1.0},
                ],
            },
        )
        self.noise_adv_classification = kwargs.pop("noise_adv_classification", False)
        self.noise_adv_regression = kwargs.pop("noise_adv_regression", False)
        self.noise_adv_loss_factor = kwargs.pop("noise_adv_loss_factor", 1.0)
        self.noise_adv_gamma = kwargs.pop("noise_adv_gamma", 10.0)
        self.representation_matching = kwargs.pop("representation_matching", None)
        # transfer
        self.freeze = kwargs.pop("freeze", None)
        self.freeze_bn = kwargs.pop("freeze_bn", {'last_layer': -1})
        self.readout_name = kwargs.pop("readout_name", "fc")
        self.reset_linear = kwargs.pop("reset_linear", False)
        self.reset_linear_frequency = kwargs.pop("reset_linear_frequency", None)
        self.transfer_from_path = kwargs.pop("transfer_from_path", None)
        self.rdm_transfer = kwargs.pop("rdm_transfer", False)
        self.rdm_prediction = kwargs.pop("rdm_prediction", {})
        self.lottery_ticket = kwargs.pop("lottery_ticket", {})
        if self.lottery_ticket:
            self.max_iter = self.lottery_ticket.get(
                "rounds", 1
            ) * self.lottery_ticket.get("round_length", 100)
        self.update(**kwargs)

    @property
    def main_loop_modules(self):
        modules = []
        if self.representation_matching:
            modules.append("RepresentationMatching")
        elif (
            self.noise_snr or self.noise_std
        ):  # Logit matching includes noise augmentation
            modules.append("NoiseAugmentation")
        if self.noise_adv_classification or self.noise_adv_regression:
            modules.append("NoiseAdvTraining")
        if self.reset_linear_frequency:
            modules.append("RandomReadoutReset")
        if self.rdm_transfer:
            modules.append("RDMPrediction")
        if self.lottery_ticket:
            modules.append("LotteryTicketPruning")
        if (
            self.rdm_transfer
            or self.noise_adv_classification
            or self.noise_adv_regression
            or self.representation_matching
        ):
            modules.append("OutputSelector")
        modules.append("ModelWrapper")
        return modules
