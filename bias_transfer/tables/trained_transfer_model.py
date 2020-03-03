from nnfabrik.main import *
from nnfabrik.template import *
from .trained_model import *
from .collapse import Collapsed
import numpy as np


@schema
class TrainedTransferModel(TrainedModelBase):
    transfer_steps = 1

    @property
    def definition(self):
        definition = """
        -> CollapsedTrained{transfer}Model{steps_m}  
        -> Model
        -> Dataset
        -> Trainer
        -> Seed
        ---
        comment='':                        varchar(768) # short description 
        score:                             float        # loss
        output:                            longblob     # trainer object's output
        ->[nullable] self.user_table
        trainedmodel_ts=CURRENT_TIMESTAMP: timestamp    # UTZ timestamp at time of insertion
        """.format(steps_m=self.transfer_steps - 1 if self.transfer_steps > 2 else "",
                   transfer="Transfer" if self.transfer_steps > 1 else "")
        return definition

    class ModelStorage(dj.Part):
        storage = 'minio'

        @property
        def definition(self):
            definition = """
            # Contains the paths to the stored models
            -> master
            ---
            model_state:            attach@{storage}
            """.format(storage=self.storage)
            return definition

    class GitLog(dj.Part):
        definition = """
        ->master
        ---
        info :              longblob
        """

    def get_full_config(self, key=None, include_state_dict=True, include_trainer=True):
        """
        Returns the full configuration dictionary needed to build all components of the network
        training including dataset, model and trainer. The returned dictionary is designed to be
        passed (with dictionary expansion) into the get_all_parts function provided in builder.py.

        Args:
            key - specific key against which to retrieve all configuration. The key must restrict all component
                  tables into a single entry. If None, will assume that this table is already restricted and
                  will obtain an existing single entry.
            include_state_dict (bool) : If True, and if key refers to a model already trained with a corresponding entry in self.ModelStorage,
                  the state_dict of the trained model is retrieved and returned
            include_trainer (bool): If False, then trainer configuration is skipped. Usually desirable when you want to simply retrieve trained model.
        """
        ret = super().get_full_config(key, include_state_dict, include_trainer)
        name = "Trained{transfer}Model{steps_m}".format(
            steps_m=self.transfer_steps - 1 if self.transfer_steps > 2 else "",
            transfer="Transfer" if self.transfer_steps > 1 else "")
        previous_training_key = globals()["Collapsed" + name]  # match collapsed key to real primary keys
        if self.transfer_steps > 1:
            previous_training = globals()[name]().ModelStorage.proj("model_state", collapsed_key_="collapsed_key") \
                                * previous_training_key  # combine collapsed key with model states from prev training
        else:
            previous_training = globals()[name]().ModelStorage.proj("model_state") \
                                * previous_training_key  # combine collapsed key with model states from prev training
        ret['trainer_config']['transfer_from_path'] = (previous_training &
                                                       {'collapsed_key': key["collapsed_key"]}).fetch1('model_state')
        return ret


@schema
class CollapsedTrainedTransferModel(Collapsed):
    Source = TrainedTransferModel()


@schema
class TrainedTransferModel2(TrainedTransferModel):
    transfer_steps = 2

    class ModelStorage(dj.Part):
        storage = 'minio'

        @property
        def definition(self):
            definition = """
              # Contains the paths to the stored models
              -> master
              ---
              model_state:            attach@{storage}
              """.format(storage=self.storage)
            return definition

    class GitLog(dj.Part):
        definition = """
          ->master
          ---
          info :              longblob
          """


@schema
class CollapsedTrainedTransferModel2(Collapsed):
    Source = TrainedTransferModel2()
