import collections

from bias_transfer.utils import stringify


class Config:
    SubConfig = collections.namedtuple("SubConfig", "config fn comment")

    defaults = {
        "dataset": "CIFAR100",
        "apply_data_augmentation": True,
        "apply_data_normalization": False,
        "optimizer": "SGD",
        "batch_size": 128,
        "num_epochs": 200,
        "lr": None,
        "lr_milestones": (
            60,
            120,
            160),
        "lr_decay": 0.7,
        "weight_decay": 5e-4,
        "momentum": 0.9,
        "add_noise": False,
        "noise_std": None,
        "noise_snr": None,
        "noise_test": {
            "noise_snr": [{5.0: 1.0}, {4.0: 1.0}, {3.0: 1.0}, {2.0: 1.0}, {1.0: 1.0}, {0.5: 1.0}, {0.0: 1.0}],
            "noise_std": [{0.0: 1.0}, {0.05: 1.0}, {0.1: 1.0}, {0.2: 1.0}, {0.3: 1.0}, {0.5: 1.0}, {1.0: 1.0}]
        },
        "transfer": False,
        "freeze": False,
        "force_cpu": False,
        "reset_linear": False,
        "reset_linear_frequency": False,
        "clean_noisy_comp_regularization": False,
        "noise_adv_classification": False,
        "noise_adv_regression": False,
        "noise_adv_loss_factor": 1.0,
        "noise_adv_gamma": 10.0,
        "use_tensorboard": False,
        "seed": 42,
        "fabrikant": "Arne Nix"
    }

    def __init__(self, **kwargs):
        self.name = self.build_name(kwargs, self.defaults)

        def get(key):
            return kwargs.get(key, self.defaults.get(key))

        if get("transfer"):
            self.name += ".transfer"
        if not get("lr"):
            kwargs["lr"] = 0.0003 if get("optimizer") == "Adam" else 0.1
        self.seed = get("seed")
        self.trainer = {"force_cpu": False,
                        "num_epochs": get("num_epochs"),
                        "optimizer": get("optimizer"),
                        "lr": get("lr"),
                        "lr_decay": get("lr_decay"),
                        "weight_decay": get("weight_decay"),
                        "momentum": get("momentum"),
                        "lr_milestones": get("lr_milestones"),
                        "add_noise": get("add_noise"),
                        "noise_std": get("noise_std"),
                        "noise_snr": get("noise_snr"),
                        "noise_test": get("noise_test"),
                        "freeze": get("freeze"),
                        "reset_linear": get("reset_linear"),
                        "noise_adv_classification": get("noise_adv_classification"),
                        "noise_adv_regression": get("noise_adv_regression"),
                        "noise_adv_loss_factor": get("noise_adv_loss_factor"),
                        "noise_adv_gamma": get("noise_adv_gamma"),
                        "use_tensorboard": get("use_tensorboard"),
                        }
        if get("clean_noisy_comp_regularization"):
            self.trainer["clean_noisy_comp_regularization"] = get("clean_noisy_comp_regularization")
        if get("reset_linear_frequency") is not False:
            self.trainer["reset_linear_frequency"] = get("reset_linear_frequency")
        self.trainer["comment"] = self.build_name(self.trainer, self.defaults)
        self.trainer_comment = self.trainer["comment"]
        self.dataset = {"dataset_cls": get("dataset"),
                     "batch_size": get("batch_size"),
                     "apply_augmentation": get("apply_data_augmentation"),
                     "apply_normalization": get("apply_data_normalization")}
        self.model = {"noise_adv_classification": get("noise_adv_classification"),
                      "noise_adv_regression": get("noise_adv_regression"),
                      "type": 50
                      }
        if get("dataset") == "CIFAR100":
            self.dataset["train_data_mean"] = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
            self.dataset["train_data_std"] = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
            self.model["num_classes"] = 100
        else:
            self.dataset["train_data_mean"] = (0.49139968, 0.48215841, 0.44653091)
            self.dataset["train_data_std"] = (0.24703223, 0.24348513, 0.26158784)
            self.model["num_classes"] = 10
        self.dataset_comment = "{}.{}".format(get("dataset"), get("batch_size"))
        self.model_comment = "resnet50.{}".format(self.model["num_classes"])
        if get("noise_adv_regression"):
            self.model_comment += ".noise_adv_regression"
        if get("noise_adv_classification"):
            self.model_comment += ".noise_adv_classification"
        self.trainer_fn = "bias_transfer.trainer.trainer"
        self.model_fn = "bias_transfer.models.resnet_builder"
        self.dataset_fn = "bias_transfer.dataset.dataset_loader"
        self.fabrikant = get("fabrikant")


    @property
    def combined_config(self):
        combined = dict(self.trainer, **self.dataset)
        combined = dict(combined, **self.model)
        return combined

    def __getitem__(self, key):
        return self.SubConfig(config=self.__getattribute__(key), fn=self.__getattribute__(key + "_fn"),
                              comment=self.__getattribute__(key + "_comment"))

    def build_name(self, settings, defaults):
        name = []
        for key, value in settings.items():
            if value is not defaults.get(key) and not (key == "reset_linear_frequency" and not value): #TODO remove this hack!
                name.append("{}_{}".format(key, stringify(value)))
        return ".".join(name)
