from nntransfer.configs.trainer.base import TrainerConfig
from nntransfer.tables.nnfabrik import Trainer


class CoTrainerConfig(TrainerConfig):
    config_name = "trainer"
    table = Trainer()
    fn = "neural_cotraining.trainer.cotrainer"

    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.force_cpu : bool = False
        self.use_tensorboard : bool = False
        self.optimizer : str = "Adam"
        self.optimizer_options : dict = {"amsgrad": False, "lr": 0.0003, "weight_decay": 5e-4}
        self.specific_opt_options : dict = {}
        self.lr_decay : float = 0.8
        self.epoch : int = 0
        self.scheduler : str = ""
        self.scheduler_options : dict = {}
        self.patience : int = 10
        self.threshold : float = 0.0001
        self.verbose : bool = False
        self.min_lr : float = 0.0001  # lr scheduler min learning rate
        self.threshold_mode : str = "rel"
        self.train_cycler : str = "LongCycler"
        self.train_cycler_args : dict = {"grad_accum_step": 0}
        self.loss_functions : dict = {"img_classification": "CrossEntropyLoss"}
        self.loss_sum : bool = False
        self.loss_weighing : bool = False
        if (
            len(self.loss_functions) > 1
            or "img_classification" not in self.loss_functions.keys()
        ):
            self.threshold_mode = "abs"
            self.scale_loss : bool = True
            self.avg_loss : bool = False

        self.maximize : bool = True # if stop_function maximized or minimized

        self.mtl : bool = False

        self.hash : bool = False

        self.return_classification_subset : int = -1

        self.to_monitor : list = ["img_classification"]

        self.interval : int = 1 # interval at which objective evaluated for early stopping
        self.max_iter : int = 100 # maximum number of iterations (epochs)

        self.restore_best : bool = True # in case of loading best model at the end of training
        self.lr_decay_steps : int = 3 # Number of times the learning rate should be reduced before stopping the training.

        self.add_final_train_eval : bool = True
        self.add_final_val_eval : bool = True
        self.add_final_test_eval : bool = True
        self.track_training : bool = False
        # noise
        self.add_noise : bool = False
        self.noise_std : dict = None
        self.noise_snr : dict = None
        self.noise_test : dict = {}
            # {
            #     "noise_snr": [
            #         {5.0: 1.0},
            #         {4.0: 1.0},
            #         {3.0: 1.0},
            #         {2.0: 1.0},
            #         {1.0: 1.0},
            #         {0.5: 1.0},
            #         {0.0: 1.0},
            #     ],
            #     "noise_std": [
            #         {0.0: 1.0},
            #         {0.05: 1.0},
            #         {0.1: 1.0},
            #         {0.2: 1.0},
            #         {0.3: 1.0},
            #         {0.5: 1.0},
            #         {1.0: 1.0},
            #     ],
            # },
        self.noise_adv_classification : bool = False
        self.noise_adv_regression : bool = False
        self.noise_adv_loss_factor : float = 1.0
        self.noise_adv_gamma : float = 10.0
        self.representation_matching : dict = {}
        # transfer
        self.freeze : dict = {}
        self.freeze_bn : dict = {'last_layer': -1}
        self.readout_name : str = "fc"
        self.reset_linear : bool = False
        self.reset_linear_frequency : bool = False
        self.transfer_from_path : str = ""
        self.rdm_transfer : bool = False
        self.rdm_prediction : dict = {}
        self.lottery_ticket : dict = {}
        if self.lottery_ticket:
            self.max_iter = self.lottery_ticket.get(
                "rounds", 1
            ) * self.lottery_ticket.get("round_length", 100)
        super().__init__(**kwargs)

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
