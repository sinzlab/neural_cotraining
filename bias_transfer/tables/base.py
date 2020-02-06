from nnfabrik.main import *
from nnfabrik.template import *
import numpy as np


@schema
class ConfigToTrainAndTransfer(dj.Manual):
    transfer_steps = 1

    @property
    def definition(self):
        definition = """
        -> Trainer
        -> Dataset
        -> Model
        -> Seed
        """

        for i in range(1, self.transfer_steps + 1):
            definition += """
            -> Model.proj(transfer_{i}_model_fn='model_fn', transfer_{i}_model_hash='model_hash')
            -> Dataset.proj(transfer_{i}_dataset_fn='dataset_fn', transfer_{i}_dataset_hash='dataset_hash')
            -> Trainer.proj(transfer_{i}_trainer_fn='trainer_fn', transfer_{i}_trainer_hash='trainer_hash')
            -> Seed.proj(transfer_{i}_seed='seed')
            """.format(i=i)

        definition += """
        ---
        config:                 longblob        # configuration object
        """

        for i in range(1, self.transfer_steps + 1):
            definition += """
            transfer_{i}_config:        longblob        # configuration object
            """.format(i=i)

        definition += """
        -> Fabrikant.proj(config_fabrikant='fabrikant_name')
        comment='' :            varchar(768)     # short description
        ts=CURRENT_TIMESTAMP:   timestamp       # UTZ timestamp at time of insertion
        """
        return definition

    def add_entry(self, configs):
        """
        inserts one new entry into the Config Table
        config -- Config object containing trainer, dataset and model options
        """
        assert len(configs) == (self.transfer_steps + 1)
        key = {}
        config = configs[-1]
        i = len(configs) - 1
        prefix = "transfer_{i}_".format(i=i) if i > 0 else ""
        for component, table in (("dataset", Dataset()), ("model", Model()), ("trainer", Trainer())):
            hash = make_hash(config[component].config)
            if not (table & (component + '_hash = "' + hash + '"')):  # not inserted yet
                table.add_entry(config[component].fn,
                                config[component].config,
                                config.fabrikant,
                                config[component].comment)
            inserted = (table & (component + '_hash = "' + hash + '"')).fetch(component + '_fn',
                                                                              component + '_hash',
                                                                              as_dict=True)[0]
            inserted = {prefix + k: v for k, v in inserted.items()}
            key.update(inserted)
        Seed().insert1({"seed": config.seed}, skip_duplicates=True)
        key[prefix + "seed"] = config.seed
        key[prefix + "config"] = config.combined_config
        if i > 0:
            if i > 2:
                sub_config_table = globals()["ConfigToTrainAndTransfer{}".format(i)]()
            elif i == 2:
                sub_config_table = globals()["ConfigToTrainAndTransfer"]()
            elif i == 1:
                sub_config_table = globals()["ConfigToTrain"]()
            key.update(sub_config_table.add_entry(configs[:-1]))
        key["comment"] = "\n".join([c.name for c in configs])
        key["config_fabrikant"] = configs[0].fabrikant
        self.insert1(key, skip_duplicates=True)
        return key


@schema
class ConfigToTrain(ConfigToTrainAndTransfer):
    transfer_steps = 0


@schema
class ConfigToTrainAndTransfer2(ConfigToTrainAndTransfer):
    transfer_steps = 2


@schema
class TrainedModelProduct(TrainedModelBase):
    table_comment = "My Trained models"


# @schema
# class ConfigToTrain(dj.Manual):
#     definition = """
#     -> Trainer
#     -> Dataset
#     -> Model
#     -> Seed
#     ---
#     config:                 longblob        # configuration object
#     -> Fabrikant.proj(config_fabrikant='fabrikant_name')
#     comment='' :            varchar(768)     # short description
#     ts=CURRENT_TIMESTAMP:   timestamp       # UTZ timestamp at time of insertion
#     """
#
#     def add_entry(self, config):
#         """
#         inserts one new entry into the Config Table
#         config -- Config object containing trainer, dataset and model options
#         """
#         key = {"config": config.combined_config,
#                "comment": config.name,
#                "seed": config.seed,
#                "config_fabrikant": config.fabrikant}
#         for component, table in (("dataset", Dataset()), ("model", Model()), ("trainer", Trainer())):
#             hash = make_hash(config[component].config)
#             if not (table & (component + '_hash = "' + hash + '"')):  # not inserted yet
#                 table.add_entry(config[component].fn,
#                                 config[component].config,
#                                 config.fabrikant,
#                                 config[component].comment)
#             inserted = (table & (component + '_hash = "' + hash + '"')).fetch(component + '_fn',
#                                                                               component + '_hash',
#                                                                               as_dict=True)[0]
#             key.update(inserted)
#         Seed().insert1({"seed": config.seed}, skip_duplicates=True)
#         self.insert1(key, skip_duplicates=True)


@schema
class TrainedModel(TrainedModelBase):
    table_comment = "My Trained models"
    comment_delimitter = '.'
    definition = """
        -> ConfigToTrain
        ---
        comment='' : varchar(768)  # short description 
        score:   float  # loss
        output: longblob  # trainer object's output
        ->[nullable] Fabrikant
        trainedmodel_ts=CURRENT_TIMESTAMP: timestamp    # UTZ timestamp at time of insertion
        """

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


@schema
class TrainedTransferModel(TrainedModelBase):
    transfer_steps = 1
    table_comment = "My Trained models"
    comment_delimitter = '\n'

    @property
    def definition(self):
        definition = """
        -> ConfigToTrainAndTransfer{steps}
        -> Trained{transfer}Model{steps_m}  # The pretrained model (potentially already product of transfer
        ---
        comment='' : varchar(768)  # short description 
        score:   float  # loss
        output: longblob  # trainer object's output
        ->[nullable] Fabrikant
        trainedmodel_ts=CURRENT_TIMESTAMP: timestamp    # UTZ timestamp at time of insertion
        """.format(steps=self.transfer_steps if self.transfer_steps > 1 else "",
                   steps_m=self.transfer_steps - 1 if self.transfer_steps > 2 else "",
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
        if key is None:
            key = self.fetch1('KEY')

        model_fn, model_config = (Model.proj(**{'transfer_{}_model_fn'.format(self.transfer_steps): 'model_fn',
                                                'transfer_{}_model_config'.format(self.transfer_steps): 'model_config',
                                                'transfer_{}_model_hash'.format(self.transfer_steps): 'model_hash'})
                                  & key).fetch1('transfer_{}_model_fn'.format(self.transfer_steps),
                                                'transfer_{}_model_config'.format(self.transfer_steps))
        dataset_fn, dataset_config = (
                Dataset.proj(**{'transfer_{}_dataset_fn'.format(self.transfer_steps): 'dataset_fn',
                                'transfer_{}_dataset_hash'.format(
                                    self.transfer_steps): 'dataset_hash',
                                'transfer_{}_dataset_config'.format(
                                    self.transfer_steps): 'dataset_config'})
                & key).fetch1('transfer_{}_dataset_fn'.format(self.transfer_steps),
                              'transfer_{}_dataset_config'.format(self.transfer_steps))
        trainer_fn, trainer_config = (
                Trainer.proj(**{'transfer_{}_trainer_fn'.format(self.transfer_steps): 'trainer_fn',
                                'transfer_{}_trainer_hash'.format(
                                    self.transfer_steps): 'trainer_hash',
                                'transfer_{}_trainer_config'.format(
                                    self.transfer_steps): 'trainer_config'})
                & key).fetch1('transfer_{}_trainer_fn'.format(self.transfer_steps),
                              'transfer_{}_trainer_config'.format(self.transfer_steps))

        for conf in (model_config, dataset_config, trainer_config):
            for k, v in conf.items():
                if type(v) == np.int64:
                    conf[k] = int(v)

        ret = dict(model_fn=model_fn, model_config=model_config,
                   dataset_fn=dataset_fn, dataset_config=dataset_config,
                   trainer_fn=trainer_fn, trainer_config=trainer_config)

        # if trained model exist and include_state_dict is True
        if include_state_dict and (self.ModelStorage & key):
            with tempfile.TemporaryDirectory() as temp_dir:
                state_dict_path = (self.ModelStorage & key).fetch1('model_state', download_path=temp_dir)
                ret['state_dict'] = torch.load(state_dict_path)

        trained_model = globals()["Trained{transfer}Model{steps_m}".format(
            steps_m=self.transfer_steps - 1 if self.transfer_steps > 2 else "",
            transfer="Transfer" if self.transfer_steps > 1 else "")]
        ret['trainer_config']['transfer_from_path'] = (trained_model().ModelStorage & key).fetch1('model_state')

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
        dataloaders, model, trainer = self.load_model(key, include_trainer=True, include_state_dict=False,
                                                      seed=seed)

        # define callback with pinging
        def call_back(**kwargs):
            self.connection.ping()
            self.call_back(**kwargs)

        # model training
        score, output, model_state = trainer(model, dataloaders, seed=seed, uid=key, cb=call_back)

        with tempfile.TemporaryDirectory() as temp_dir:
            filename = make_hash(key) + '.pth.tar'
            filepath = os.path.join(temp_dir, filename)
            torch.save(model_state, filepath)

            key['score'] = score
            key['output'] = output
            key['fabrikant_name'] = fabrikant_name
            comments = []
            comments.append((self.trainer_table & key).fetch1("trainer_comment"))
            comments.append((self.model_table & key).fetch1("model_comment"))
            comments.append((self.dataset_table & key).fetch1("dataset_comment"))
            for s in range(1, self.transfer_steps + 1):
                comments.append((Trainer.proj(**{'transfer_{}_trainer_fn'.format(s): 'trainer_fn',
                                                 'transfer_{}_trainer_hash'.format(s): 'trainer_hash',
                                                 'transfer_{}_trainer_comment'.format(
                                                     s): 'trainer_comment'})
                                 & key).fetch1("transfer_{}_trainer_comment".format(s)))
                comments.append((Model.proj(**{'transfer_{}_model_fn'.format(s): 'model_fn',
                                               'transfer_{}_model_hash'.format(s): 'model_hash',
                                               'transfer_{}_model_comment'.format(s): 'model_comment'})
                                 & key).fetch1("transfer_{}_model_comment".format(s)))
                comments.append((Dataset.proj(**{'transfer_{}_dataset_fn'.format(s): 'dataset_fn',
                                                 'transfer_{}_dataset_hash'.format(s): 'dataset_hash',
                                                 'transfer_{}_dataset_comment'.format(
                                                     s): 'dataset_comment'})
                                 & key).fetch1("transfer_{}_dataset_comment".format(s)))
            key['comment'] = self.comment_delimitter.join(comments)
            self.insert1(key)

            key['model_state'] = filepath

            self.ModelStorage.insert1(key, ignore_extra_fields=True)


@schema
class TrainedTransferModel2(TrainedTransferModel):
    transfer_steps = 2
    table_comment = "My Trained models"
    comment_delimitter = '\n'

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
