from .config import Config
from .dataset import *
from .model import *
from .trainer import *

import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class Experiment(Config):
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
    def from_dict(cls, config_dicts: list) -> "Experiment":
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
        configs = []
        for c_dict in config_dicts:
            configs.append(Config.from_dict(c_dict))
        return cls(configs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary.

        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        return [c.to_dict() for c in self.configs]
