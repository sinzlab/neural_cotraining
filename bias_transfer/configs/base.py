import os
import json
import copy
import logging
from typing import Dict

from nnfabrik.utility.dj_helpers import make_hash

logger = logging.getLogger(__name__)


class BaseConfig(object):
    r""" Base class for all configuration classes.
        Handles methods for loading/saving configurations, and interaction with nnfabrik.

        Adapted from https://github.com/huggingface/transformers/blob/master/src/transformers/configuration_utils.py
    """

    config_name = ""
    table = None
    fn = None

    def __init__(self, **kwargs):
        self.description = kwargs.pop("description")
        if self.description == "Default":
            self.description = ""

    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            if name == "__getstate__" or name == "__setstate__":  # for deepcopy to work
                raise AttributeError
            else:
                return None

    def update(self, **kwargs):
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error("Can't set {} with value {} for {}".format(key, value, self))
                raise err

    def get_key(self):
        hash = make_hash(self.to_dict())
        return {self.config_name + '_hash': hash,
                self.config_name + '_fn': self.fn}

    def add_to_table(self):
        """
        Insert the config (+ fn and comment) into the dedicated table if not present already
        :return:
        """
        assert self.table is not None and self.fn is not None
        if not (self.table & self.get_key()):
            self.table.add_entry(self.fn,
                                 self.to_dict(),
                                 None,  # Fabrikant will automatically be set to current user
                                 self.description)

    def save(self, save_directory):
        """
        Save a configuration object to the directory `save_directory`, so that it
        can be re-loaded using the `load` class method.

        Args:
            save_directory (:obj:`string`):
                Directory where the configuration JSON file will be saved.
        """
        assert os.path.isdir(
            save_directory
        ), "Saving path should be a directory where the model and configuration can be saved"

        # If we save using the predefined names, we can load using `from_pretrained`
        output_config_file = os.path.join(save_directory, self.config_name)

        self.to_json_file(output_config_file)
        logger.info("Configuration saved in {}".format(output_config_file))

    @classmethod
    def from_dict(cls, config_dict: Dict, **kwargs) -> "BaseConfig":
        """
        Constructs a `BaseConfig` from a Python dictionary of parameters.

        Args:
            config_dict (:obj:`Dict[str, any]`):
                Dictionary that will be used to instantiate the configuration object. Such a dictionary can be retrieved
                from a pre-trained checkpoint by leveraging the :func:`~transformers.PretrainedConfig.get_config_dict`
                method.
            kwargs (:obj:`Dict[str, any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            :class:`BaseConfig`: An instance of a configuration object
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        config = cls(**config_dict)

        # Update config with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        # logger.info("Config %s", str(config))
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    @classmethod
    def from_json_file(cls, json_file: str) -> "BaseConfig":
        """
        Constructs a `BaseConfig` from the path to a json file of parameters.

        Args:
            json_file (:obj:`string`):
                Path to the JSON file containing the parameters.

        Returns:
            :class:`BaseConfig`: An instance of a configuration object

        """
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        config_dict = json.loads(text)
        return cls.from_dict(config_dict)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return "{} {}".format(self.__class__.__name__, self.to_json_string())

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary.

        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.

        Returns:
            :obj:`string`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """
        Save this instance to a json file.

        Args:
            json_file_path (:obj:`string`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())
