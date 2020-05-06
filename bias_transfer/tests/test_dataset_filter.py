import unittest
from bias_transfer.configs import dataset
from bias_transfer.tests._base import BaseTest
import nnfabrik as nnf


class DatasetFilterTest(BaseTest):
    def test_cifar100(self):
        print("===================================================", flush=True)
        print("=================TEST CIFAR100 Filter==============", flush=True)
        start = 10
        end = 90
        dataset_conf = dataset.ImageDatasetConfig(
            comment="Minimal CIFAR100",
            dataset_cls="CIFAR100",
            apply_data_normalization=True,
            apply_data_augmentation=True,
            add_corrupted_test=True,
            valid_size=0.00,
            filter_classes=(start, end),
            seed=42,
        )
        data_loaders = nnf.builder.get_data(dataset_conf.fn, dataset_conf.to_dict())
        self.assertEqual(len(data_loaders["train"]["img_classification"].dataset), (end-start) * 500)
        self.assertEqual(len(data_loaders["test"]["img_classification"].dataset), (end-start) * 100)
        self.assertEqual(len(data_loaders["c_test"]["frost"][1].dataset), (end-start) * 100)


    def test_cifar10(self):
        print("===================================================", flush=True)
        print("=================TEST CIFAR10 Filter===============", flush=True)
        start = 2
        end = 10
        dataset_conf = dataset.ImageDatasetConfig(
            comment="Minimal CIFAR10",
            dataset_cls="CIFAR10",
            apply_data_normalization=True,
            apply_data_augmentation=True,
            add_corrupted_test=True,
            valid_size=0.00,
            filter_classes=(start, end),
            seed=42,
        )
        data_loaders = nnf.builder.get_data(dataset_conf.fn, dataset_conf.to_dict())
        self.assertEqual(len(data_loaders["train"]["img_classification"].dataset), (end-start) * 5000)
        self.assertEqual(len(data_loaders["test"]["img_classification"].dataset), (end-start) * 1000)
        self.assertEqual(len(data_loaders["c_test"]["speckle_noise"][5].dataset), (end-start) * 1000)


    def test_tiny_imagenet(self):
        print("===================================================", flush=True)
        print("===============TEST TinyImageNet Filter============", flush=True)
        start = 0
        end = 150
        dataset_conf = dataset.ImageDatasetConfig(
            comment="Minimal TinyImageNet",
            dataset_cls="TinyImageNet",
            apply_data_normalization=True,
            apply_data_augmentation=True,
            add_corrupted_test=True,
            valid_size=0.00,
            filter_classes=(start, end),
            seed=42,
        )
        data_loaders = nnf.builder.get_data(dataset_conf.fn, dataset_conf.to_dict())
        self.assertEqual(len(data_loaders["train"]["img_classification"].dataset), (end-start) * 500)
        self.assertEqual(len(data_loaders["test"]["img_classification"].dataset), (end-start) * 50)
        self.assertEqual(len(data_loaders["c_test"]["snow"][5].dataset), (end-start) * 50)

if __name__ == "__main__":
    unittest.main()
