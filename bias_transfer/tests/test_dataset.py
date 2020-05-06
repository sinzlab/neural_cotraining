import unittest
import os
from bias_transfer.configs import trainer, model, dataset
from bias_transfer.tests._base import BaseTest
import nnfabrik as nnf


class DatasetTest(BaseTest):
    def test_cifar100(self):
        print("===================================================", flush=True)
        print("================TEST CIFAR100 Training=============", flush=True)
        model_conf = model.ClassificationModelConfig(
            comment="CIFAR100", dataset_cls="CIFAR100", type="resnet18",
        )
        dataset_conf = dataset.ImageDatasetConfig(
            comment="Minimal CIFAR100",
            dataset_cls="CIFAR100",
            apply_data_normalization=True,
            apply_data_augmentation=True,
            add_corrupted_test=True,
            valid_size=0.95,
        )
        self.get_parts(dataset_conf, model_conf, self.seed)
        trainer_conf = trainer.TrainerConfig(
            comment="CIFAR100 Training Test",
            max_iter=2,
            verbose=False,
            add_noise=False,
            noise_snr=None,
            noise_std=None,
            noise_test={"noise_snr": [], "noise_std": [],},
            restore_best=False,
            early_stop=False,
            lr_milestones=(1,),
            adaptive_lr=False,
            patience=1000,
        )
        score = self.run_training(trainer_conf)
        self.assertAlmostEqual(score, 9.92, places=1)

    def test_tiny_imagenet(self):
        print("===================================================", flush=True)
        print("==============TEST Tiny-ImageNet Training==========", flush=True)
        model_conf = model.ClassificationModelConfig(
            comment="TinyImageNet", dataset_cls="TinyImageNet", type="resnet18",
        )
        dataset_conf = dataset.ImageDatasetConfig(
            comment="Minimal TinyImageNet",
            dataset_cls="TinyImageNet",
            apply_data_normalization=True,
            apply_data_augmentation=True,
            add_corrupted_test=True,
            valid_size=0.95,
        )
        self.get_parts(dataset_conf, model_conf, self.seed)
        trainer_conf = trainer.TrainerConfig(
            comment="TinyImageNet Training Test",
            max_iter=2,
            verbose=False,
            add_noise=False,
            noise_snr=None,
            noise_std=None,
            noise_test={"noise_snr": [], "noise_std": [],},
            restore_best=False,
            early_stop=False,
            lr_milestones=(1,),
            adaptive_lr=False,
            patience=1000,
        )
        score = self.run_training(trainer_conf)
        self.assertAlmostEqual(score, 5.52, places=1)

    def test_imagenet(self):
        print("===================================================", flush=True)
        print("=================TEST ImageNet Training============", flush=True)
        model_conf = model.ClassificationModelConfig(
            comment="ImageNet", dataset_cls="ImageNet", type="resnet18",
        )
        dataset_conf = dataset.ImageDatasetConfig(
            comment="Minimal ImageNet",
            dataset_cls="ImageNet",
            apply_data_normalization=True,
            apply_data_augmentation=True,
            add_corrupted_test=True,
            batch_size=70,
            valid_size=0.995,
        )
        self.get_parts(dataset_conf, model_conf, self.seed)
        trainer_conf = trainer.TrainerConfig(
            comment="ImageNet Training Test",
            max_iter=1,
            verbose=False,
            add_noise=False,
            noise_snr=None,
            noise_std=None,
            noise_test={"noise_snr": [], "noise_std": [],},
            restore_best=False,
            lr_milestones=(1,),
            adaptive_lr=False,
            early_stop=False,
            patience=1000,
        )
        score = self.run_training(trainer_conf)
        self.assertAlmostEqual(score, 0.9990633780830471, places=1)

    def test_imagenet_pretrained(self):
        print("===================================================", flush=True)
        print("================TEST ImageNet Pretrained===========", flush=True)
        model_conf = model.ClassificationModelConfig(
            comment="ImageNet", dataset_cls="ImageNet", type="resnet50", pretrained=True
        )
        dataset_conf = dataset.ImageDatasetConfig(
            comment="Minimal ImageNet",
            dataset_cls="ImageNet",
            apply_data_normalization=True,
            apply_data_augmentation=True,
            add_corrupted_test=False,
            valid_size=0.01,
        )
        self.data_loaders, self.model = nnf.builder.get_all_parts(
            dataset_fn=dataset_conf.fn,
            dataset_config=dataset_conf.to_dict(),
            model_fn=model_conf.fn,
            model_config=model_conf.to_dict(),
            seed=self.seed,
            trainer_fn=None,
            trainer_config=None,
        )
        if "c_test" in self.data_loaders:
            category_1 = list(self.data_loaders["c_test"].keys)[0]
            self.data_loaders["c_test"] = {
                category_1: {1: self.data_loaders["c_test"][category_1][1]}
            }
        trainer_conf = trainer.TrainerConfig(
            comment="ImageNet Training Test",
            max_iter=0,
            verbose=False,
            add_noise=False,
            noise_snr=None,
            noise_std=None,
            noise_test={"noise_snr": [], "noise_std": [],},
            restore_best=False,
            lr_milestones=(1,),
            adaptive_lr=False,
            early_stop=False,
            patience=1000,
        )
        uid = "test1"
        path = "./checkpoint/ckpt.{}.pth".format(nnf.utility.dj_helpers.make_hash(uid))
        if os.path.exists(path):
            os.remove(path)

        trainer_fn = nnf.builder.get_trainer(trainer_conf.fn, trainer_conf.to_dict())

        def call_back(**kwargs):
            pass

        # model training
        score, output, model_state = trainer_fn(
            model=self.model,
            dataloaders=self.data_loaders,
            seed=self.seed,
            uid=uid,
            cb=call_back,
        )
        self.assertAlmostEqual(score, 76.1, places=1)

    def test_imagenet_pretrained_vgg(self):
        print("===================================================", flush=True)
        print("=============TEST ImageNet Pretrained (VGG)========", flush=True)
        model_conf = model.ClassificationModelConfig(
            comment="ImageNet", dataset_cls="ImageNet", type="vgg19_bn", pretrained=True
        )
        dataset_conf = dataset.ImageDatasetConfig(
            comment="Minimal ImageNet",
            dataset_cls="ImageNet",
            apply_data_normalization=True,
            apply_data_augmentation=True,
            add_corrupted_test=False,
            valid_size=0.01,
        )
        self.data_loaders, self.model = nnf.builder.get_all_parts(
            dataset_fn=dataset_conf.fn,
            dataset_config=dataset_conf.to_dict(),
            model_fn=model_conf.fn,
            model_config=model_conf.to_dict(),
            seed=self.seed,
            trainer_fn=None,
            trainer_config=None,
        )
        if "c_test" in self.data_loaders:
            category_1 = list(self.data_loaders["c_test"].keys)[0]
            self.data_loaders["c_test"] = {
                category_1: {1: self.data_loaders["c_test"][category_1][1]}
            }
        trainer_conf = trainer.TrainerConfig(
            comment="ImageNet Training Test",
            max_iter=0,
            verbose=False,
            add_noise=False,
            noise_snr=None,
            noise_std=None,
            noise_test={"noise_snr": [], "noise_std": [],},
            restore_best=False,
            early_stop=False,
            lr_milestones=(1,),
            adaptive_lr=False,
            patience=1000,
        )
        uid = "test1"
        path = "./checkpoint/ckpt.{}.pth".format(nnf.utility.dj_helpers.make_hash(uid))
        if os.path.exists(path):
            os.remove(path)

        trainer_fn = nnf.builder.get_trainer(trainer_conf.fn, trainer_conf.to_dict())

        def call_back(**kwargs):
            pass

        # model training
        score, output, model_state = trainer_fn(
            model=self.model,
            dataloaders=self.data_loaders,
            seed=self.seed,
            uid=uid,
            cb=call_back,
        )
        self.assertAlmostEqual(score, 74.24, places=1)

if __name__ == "__main__":
    unittest.main()
