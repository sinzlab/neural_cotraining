from nnfabrik.main import *

@schema
# @gitlog
class TrainedTransferModel(dj.Computed):
    definition = """
    -> Model.proj(transfer_model_fn='model_fn', transfer_model_hash='model_hash')
    -> Dataset.proj(transfer_dataset_fn='dataset_fn', transfer_dataset_hash='dataset_hash')
    -> Trainer.proj(transfer_trainer_fn='trainer_fn', transfer_trainer_hash='trainer_hash')
    -> Seed.proj(transfer_seed='seed')
    -> TrainedModel
    ---
    comment='' : varchar(1536)  # short description 
    score:   float  # loss
    output: longblob  # trainer object's output
    ->[nullable] Fabrikant
    trainedmodel_ts=CURRENT_TIMESTAMP: timestamp    # UTZ timestamp at time of insertion
    """

    def get_full_config(self, key=None, include_state_dict=True):
        if key is None:
            key = self.fetch1('KEY')

        model_fn, model_config = (Model & key).fn_config
        dataset_fn, dataset_config = (Dataset & key).fn_config
        trainer_fn, trainer_config = (Trainer & key).fn_config

        ret = dict(model_fn=model_fn, model_config=model_config,
                   dataset_fn=dataset_fn, dataset_config=dataset_config,
                   trainer_fn=trainer_fn, trainer_config=trainer_config)

        # if trained model exist and include_state_dict is True
        if include_state_dict and (self & key):
            ret['state_dict'] = (self.ModelStorage & key).fetch1('model_state')

        ret['trainer_config']['transfer_from_path'] = (TrainedModel().ModelStorage & key).fetch1('model_state')

        return ret

    class ModelStorage(dj.Part):
        definition = """
        # Contains the paths to the stored models
        -> master
        ---
        model_state:            attach@minio
        """

    class GitLog(dj.Part):
        definition = """
        ->master
        ---
        info :              longblob
        """

    def get_entry(self, key):
        (Dataset & key).fetch()

    def make(self, key):

        commits_info = {name: info for name, info in [check_repo_commit(repo) for repo in config['repos']]}
        assert len(commits_info) == len(config['repos'])

        if any(['error_msg' in name for name in commits_info.keys()]):
            err_msgs = ["You have uncommited changes."]
            err_msgs.extend([info for name, info in commits_info.items() if 'error_msg' in name])
            err_msgs.append("\nPlease commit the changes before running populate.\n")
            raise RuntimeError('\n'.join(err_msgs))

        else:

            # by default try to lookup the architect corresponding to the current DJ user
            fabrikant_name = Fabrikant.get_current_user()
            seed = (Seed & key).fetch1('seed')

            config_dict = self.get_full_config(key)
            dataloaders, model, trainer = get_all_parts(**config_dict, seed=seed)

            # model training
            score, output, model_state = trainer(model, seed, dataloaders)

            with tempfile.TemporaryDirectory() as trained_models:
                filename = make_hash(key) + '.pth.tar'
                filepath = os.path.join(trained_models, filename)
                torch.save(model_state, filepath)

                key['score'] = score
                key['output'] = output
                key['fabrikant_name'] = fabrikant_name
                comments = []
                comments.append((Trainer & key).fetch1("trainer_comment"))
                comments.append((Model & key).fetch1("model_comment"))
                comments.append((Dataset & key).fetch1("dataset_comment"))
                key['comment'] = '.'.join(comments)
                self.insert1(key)

                key['model_state'] = filepath
                self.ModelStorage.insert1(key, ignore_extra_fields=True)

                # add the git info to the part table
                if commits_info:
                    key['info'] = commits_info
                    self.GitLog().insert1(key, skip_duplicates=True, ignore_extra_fields=True)
