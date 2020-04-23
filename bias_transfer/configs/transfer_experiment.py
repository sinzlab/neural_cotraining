import copy

from nnfabrik.utility.dj_helpers import make_hash

from .experiment import Experiment


class TransferExperiment(Experiment):
    r""" Collection of potentially multiple configs to define an experiment
    """

    config_name = "config"
    table = None
    fn = None

    def __init__(self, configs):
        self.configs = configs
        comment = []
        for c in self.configs:
            comment.append(c.comment)
            c.comment = " -> ".join(comment)
        self.comment = " -> ".join(comment)

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
