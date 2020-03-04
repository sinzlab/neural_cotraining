from typing import Dict

from .base import BaseConfig
from nnfabrik.main import Seed


class Experiment(BaseConfig):
    r""" Wrapper class around dataset, model and trainer configs
    """

    config_name = "config"
    table = None
    fn = None

    def __init__(self, dataset, model, trainer, seed):
        self.dataset = dataset
        self.model = model
        self.trainer = trainer
        self.seed = seed
        self.description = self.trainer.description

    def get_key(self):
        key = self.dataset.get_key()
        key.update(self.model.get_key())
        key.update(self.trainer.get_key())
        key.update({"seed": self.seed})
        return key

    def get_restrictions(self):
        return [self.get_key()]

    def add_to_table(self):
        """
        Insert the config (+ fn and comment) into the dedicated table if not present already
        :return:
        """
        self.dataset.add_to_table()
        self.model.add_to_table()
        self.trainer.add_to_table()
        Seed().insert1({"seed": self.seed}, skip_duplicates=True)

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "Experiment":
        """
        Constructs a `Config` from a Python dictionary of parameters.

        Args:
            config_dict (:obj:`Dict[str, any]`):
                Dictionary that will be used to instantiate the configuration object. Such a dictionary can be retrieved
                from a pre-trained checkpoint by leveraging the :func:`~transformers.PretrainedConfig.get_config_dict`
                method.
        Returns:
            :class:`Experiment`: An instance of a configuration object
        """
        dataset_cls, dataset_dict = config_dict.get("dataset", ("DatasetConfig", {}))
        dataset = globals()[dataset_cls].from_dict(dataset_dict)
        model_cls, model_dict = config_dict.get("model", ("ModelConfig", {}))
        model = globals()[model_cls].from_dict(model_dict)
        trainer_cls, trainer_dict = config_dict.get("trainer", ("TrainerConfig", {}))
        trainer = globals()[trainer_cls].from_dict(trainer_dict)
        seed = config_dict.get("seed", 42)
        return cls(dataset, model, trainer, seed)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary.

        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = {
            "dataset": (self.dataset.__class__.__name__, self.dataset.to_dict()),
            "model": (self.model.__class__.__name__, self.model.to_dict()),
            "trainer": (self.trainer.__class__.__name__, self.trainer.to_dict()),
            "seed": self.seed
        }
        return output