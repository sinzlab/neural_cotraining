class Config:
    def __init__(self,
                 dataset: str = "CIFAR100",
                 apply_data_augmentation: bool = False,
                 apply_data_normalization: bool = False,
                 optimizer: str = "SGD",
                 batch_size: int = 128,
                 num_epochs: int = 200,
                 lr: float = None,
                 lr_milestones: tuple = (60, 120, 160),
                 lr_decay: float = 0.7,
                 add_noise: bool = False,
                 noise_std: float = None,
                 noise_snr: float = 0.9,
                 transfer: bool = False,
                 freeze: bool = False,
                 reset_linear: bool = False
                 ):
        self.name = self.build_name(dataset=dataset, apply_data_augmentation=apply_data_augmentation,
                                    apply_data_normalization=apply_data_normalization, optimizer=optimizer,
                                    batch_size=batch_size, num_epochs=num_epochs,
                                    lr=lr, lr_milestones=lr_milestones, lr_decay=lr_decay, add_noise=add_noise,
                                    noise_std=noise_std, noise_snr=noise_snr)
        if transfer:
            self.name += ".transfer"
        if not lr:
            lr = 0.0003 if optimizer == "Adam" else 0.1
        self.trainer = {"force_cpu": False,
                        "resume": False,
                        "num_epochs": num_epochs,
                        "optimizer": optimizer,
                        "lr": lr,
                        "lr_decay": lr_decay,
                        "lr_milestones": lr_milestones,
                        "add_noise": add_noise,
                        "noise_std": noise_std,
                        "noise_snr": noise_snr,
                        "freeze": freeze,
                        "reset_linear": reset_linear,
                        "comment": self.name}
        self.data = {"dataset_cls": dataset,
                     "batch_size": batch_size,
                     "apply_augmentation": apply_data_augmentation,
                     "apply_normalization": apply_data_normalization}
        if dataset == "CIFAR100":
            self.data["train_data_mean"] = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
            self.data["train_data_std"] = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
            self.model = {"num_classes": 100}
        else:
            self.data["train_data_mean"] = (0.49139968, 0.48215841, 0.44653091)
            self.data["train_data_std"] = (0.24703223, 0.24348513, 0.26158784)
            self.model = {"num_classes": 10}
        self.data_comment = "{}.{}".format(dataset, batch_size)
        self.model_comment = "resnet50.{}".format(self.model["num_classes"])

    def build_name(self, **kwargs):
        name = []
        for i, (key, value) in enumerate(kwargs.items()):
            if self.__init__.__kwdefaults__ and key in self.__init__.__kwdefaults__:
                if value is not self.__init__.__kwdefaults__[key]:
                    name.append("{}_{}".format(key, value))
            elif i < len(self.__init__.__defaults__):
                if value is not self.__init__.__defaults__[i]:
                    name.append("{}_{}".format(key, value))
            else:
                name.append("{}_{}".format(key, value))
        return ".".join(name)
