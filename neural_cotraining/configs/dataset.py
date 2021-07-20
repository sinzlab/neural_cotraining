from nntransfer.configs.dataset.base import DatasetConfig
from nntransfer.tables.nnfabrik import Dataset


class ImageDatasetConfig(DatasetConfig):
    config_name = "dataset"
    table = Dataset()
    fn = "neural_cotraining.dataset.img_dataset_loader"

    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.dataset_cls : str = "TinyImageNet"
        self.apply_augmentation : bool = True
        self.apply_normalization : bool = True
        self.apply_grayscale : bool = False
        self.apply_noise : dict = {}
        self.in_resize : int = 256
        if self.dataset_cls == "TinyImageNet":
            if self.apply_grayscale:
                self.train_data_mean = (0.4519,)
                self.train_data_std = (0.2221,)
            else:
                self.train_data_mean = (
                    0.4802,
                    0.4481,
                    0.3975,
                )
                self.train_data_std = (
                    0.2302,
                    0.2265,
                    0.2262,
                )
            self.data_dir : str = "./data/image_classification/"
            self.input_size : int = 64
            self.num_workers : int = 1
            self.valid_size : float = 0.1
        elif self.dataset_cls == "ImageNet":
            if self.apply_grayscale:
                self.train_data_mean = (0.4573,)
                self.train_data_std = (0.2156,)
            else:
                self.train_data_mean = (0.485, 0.456, 0.406)
                self.train_data_std = (0.229, 0.224, 0.225)
            self.data_dir : str = "./data/image_classification/"
            self.input_size : int = 224
            self.num_workers : int = 8
            self.valid_size : float = 0.01
        else:
            raise NameError()
        self.add_corrupted_test : bool = True
        self.add_fly_corrupted_test : dict = {'jpeg_compression': [1, 2, 3, 4, 5],
                                                'pixelate': [1, 2, 3, 4, 5],
                                                'impulse_noise': [1, 2, 3, 4, 5], 'defocus_blur': [1, 2, 3, 4, 5],
                                                'contrast': [1, 2, 3, 4, 5], 'fog': [1, 2, 3, 4, 5],
                                                'brightness': [1, 2, 3, 4, 5], 'frost': [1, 2, 3, 4, 5],
                                                'glass_blur': [1, 2, 3, 4, 5], 'shot_noise': [1, 2, 3, 4, 5],
                                                'motion_blur': [1, 2, 3, 4, 5], 'gaussian_noise': [1, 2, 3, 4, 5],
                                                'elastic_transform': [1, 2, 3, 4, 5], 'snow': [1, 2, 3, 4, 5],
                                                  'zoom_blur': [1, 2, 3, 4, 5]}

        self.add_stylized_test : bool = False
        self.shuffle : bool = True
        self.show_sample : bool = False
        self.filter_classes : tuple = None  # (start,end)
        self.pin_memory : bool = True
        super().__init__(**kwargs)

    @property
    def filters(self):
        filters = []
        if self.filter_classes:
            filters.append("ClassesFilter")
        return filters


class NeuralDatasetConfig(DatasetConfig):
    config_name = "dataset"
    table = Dataset()
    fn = "neural_cotraining.dataset.neural_loader"

    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.train_frac : float = 0.8
        self.dataset : str = "CSRF19_V1"
        self.sessions_dir : str = "neuronal_data"
        self.data_dir = "./data/monkey/toliaslab/{}".format(self.dataset)
        self.seed : int = 1000
        self.subsample : int = 1
        self.crop : int = 70
        self.time_bins_sum : int = 12
        self.target_types : list = ["v1"]
        self.normalize : bool = True
        self.train_transformation : bool = False  #for resize and grayscale
        self.individual_image_paths : bool = False
        self.stats : dict = {}
        self.pin_memory : bool = True
        self.load_all_in_memory : bool = True
        self.num_workers : int = 1
        self.apply_augmentation : bool = False
        self.input_size : int = 64
        self.apply_grayscale : bool = True
        self.add_fly_corrupted_test : dict = {'jpeg_compression': [1, 2, 3, 4, 5], 'pixelate': [1, 2, 3, 4, 5],
                                                  'impulse_noise': [1, 2, 3, 4, 5], 'defocus_blur': [1, 2, 3, 4, 5],
                                                  'contrast': [1, 2, 3, 4, 5], 'fog': [1, 2, 3, 4, 5],
                                                  'brightness': [1, 2, 3, 4, 5], 'frost': [1, 2, 3, 4, 5],
                                                  'glass_blur': [1, 2, 3, 4, 5], 'shot_noise': [1, 2, 3, 4, 5],
                                                  'motion_blur': [1, 2, 3, 4, 5], 'gaussian_noise': [1, 2, 3, 4, 5],
                                                  'elastic_transform': [1, 2, 3, 4, 5], 'snow': [1, 2, 3, 4, 5],
                                                  'zoom_blur': [1, 2, 3, 4, 5]}
        self.resize : int = 0
        super().__init__(**kwargs)


class MTLDatasetsConfig(DatasetConfig):

    '''
    This class is to combine the neural dataset and image classification set for MTL.
    Using one config for both allows us to alternate between batches in a flexible manner
    '''

    config_name = "dataset"
    table = Dataset()
    fn = "neural_cotraining.dataset.mtl_loader"

    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.classification_loader : str = "img_classification"
        self.v1_dataset_dict : dict = {}
        self.v4_dataset_dict : dict = {}
        if self.v1_dataset_dict:
            self.v1_dataset_config = NeuralDatasetConfig(
                **self.v1_dataset_dict
            ).to_dict()
        else:
            self.v1_dataset_config = {}
        if self.v4_dataset_dict:
            self.v4_dataset_config = NeuralDatasetConfig(
                **self.v4_dataset_dict
            ).to_dict()
        else:
            self.v4_dataset_config = {}
        self.img_dataset_dict : dict = {}
        if self.classification_loader == "img_classification":
            self.img_dataset_config = ImageDatasetConfig(**self.img_dataset_dict).to_dict()
        else:
            self.img_dataset_config = NeuralDatasetConfig(**self.img_dataset_dict).to_dict()

        super().__init__(**kwargs)
