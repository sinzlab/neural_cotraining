from nnfabrik.main import *
from nnfabrik.template import *
from .trained_model import TrainedModel


@schema
class EvaluatedModel(TrainedModelBase):
    table_comment = "Custom evaluation for trained models"

    definition = """
    -> TrainedModel
    ---
    ->[nullable] Fabrikant
    score:                             float        # loss
    output:                            longblob     # trainer object's output
    evaluatedmodel_ts=CURRENT_TIMESTAMP: timestamp    # UTZ timestamp at time of insertion
    """
    ModelStorage = None

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
        if key is None:
            key = self.fetch1('KEY')

        model_fn, model_config = (self.model_table & key).fn_config
        dataset_fn, dataset_config = (self.dataset_table & key).fn_config

        ret = dict(model_fn=model_fn, model_config=model_config,
                   dataset_fn=dataset_fn, dataset_config=dataset_config)

        if include_trainer:
            trainer_fn, trainer_config = (self.trainer_table & key).fn_config
            ret['trainer_fn'] = trainer_fn
            ret['trainer_config'] = trainer_config

        # if trained model exist and include_state_dict is True
        if include_state_dict and (TrainedModel.ModelStorage & key):
            with tempfile.TemporaryDirectory() as temp_dir:
                state_dict_path = (TrainedModel.ModelStorage & key).fetch1('model_state', download_path=temp_dir)
                ret['state_dict'] = torch.load(state_dict_path)

        return ret

    def make(self, key):
        """
        Given key specifying configuration for dataloaders, model and trainer,
        trains the model and saves the trained model.
        """
        # lookup the fabrikant corresponding to the current DJ user
        fabrikant_name = Fabrikant.get_current_user()
        seed = (Seed & key).fetch1('seed')

        # load everything
        dataloaders, model, trainer = self.load_model(key, include_trainer=True, include_state_dict=True, seed=seed)
        # model = ((TrainedModel() & key).ModelStorage()).fetch1("model_state")

        # define callback with pinging
        def call_back(**kwargs):
            self.connection.ping()
            self.call_back(**kwargs)

        # model training
        score, output, model_state = trainer(model, dataloaders, seed=seed, uid=key, cb=call_back, eval_only=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            filename = make_hash(key) + '.pth.tar'
            filepath = os.path.join(temp_dir, filename)
            torch.save(model_state, filepath)

            key['score'] = score
            key['output'] = output
            key['fabrikant_name'] = fabrikant_name
            self.insert1(key)
