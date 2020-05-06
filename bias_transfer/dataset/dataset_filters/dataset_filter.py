class DatasetFilter(object):
    def __init__(self, config, train_dataset):
        self.config = config

    def apply(self, dataset):
        raise NotImplementedError
