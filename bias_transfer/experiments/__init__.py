import copy
from collections import namedtuple
from typing import Dict

from nnfabrik.main import Seed
from nnfabrik.utility.dj_helpers import make_hash

from bias_transfer.configs.base import BaseConfig

Description = namedtuple("Description", "name seed")


class Experiment(BaseConfig):
    r""" Wrapper class around dataset, model and trainer configs
    """

    config_name = "config"
    table = None
    fn = None
    fabrikant = "Arne Nix"

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


class TransferExperiment(Experiment):
    r""" Collection of potentially multiple configs to define an experiment
    """

    config_name = "config"
    table = None
    fn = None
    fabrikant = "Arne Nix"

    def __init__(self, configs):
        self.configs = configs
        description = []
        for c in self.configs:
            description.append(c.description)
            c.description = " -> ".join(description)
        self.description = " -> ".join(description)

    def get_restrictions(self):
        key = []
        for i, config in enumerate(self.configs):
            sub_key = config.get_key()
            if i > 0:
                key_for_hash = copy.deepcopy(key[-1])
                if i > 1:
                    key_for_hash["collapsed_key_"] = key_for_hash["collapsed_key"]
                    del key_for_hash["collapsed_key"]
                sub_key["collapsed_key"] = make_hash(key_for_hash)
            key.append(sub_key)
        return key

    def add_to_table(self):
        """
        Insert the config (+ fn and comment) into the dedicated table if not present already
        :return:
        """
        for config in self.configs:
            config.add_to_table()

    @classmethod
    def from_dict(cls, config_dicts: list) -> "TransferExperiment":
        """
        Constructs a `Config` from a Python dictionary of parameters.

        Args:
            config_dict (:obj:`Dict[str, any]`):
                Dictionary that will be used to instantiate the configuration object. Such a dictionary can be retrieved
                from a pre-trained checkpoint by leveraging the :func:`~transformers.PretrainedConfig.get_config_dict`
                method.
        Returns:
            :class:`TransferExperiment`: An instance of a configuration object
        """
        configs = []
        for c_dict in config_dicts:
            configs.append(Experiment.from_dict(c_dict))
        return cls(configs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary.

        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        return [c.to_dict() for c in self.configs]