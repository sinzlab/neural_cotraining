import unittest
from bias_transfer.configs import trainer, model, dataset
from bias_transfer.tests._base import BaseTest
import nnfabrik as nnf


class DatasetTest(BaseTest):
    # def test_cifar100(self):
    #     print("===================================================", flush=True)
    #     print("================TEST CIFAR100 Training=============", flush=True)
    #     model_conf = model.ClassificationModelConfig(
    #         comment="CIFAR100", dataset_cls="CIFAR100", type="18",
    #     )
    #     dataset_conf = dataset.ImageDatasetConfig(
    #         comment="Minimal CIFAR100",
    #         dataset_cls="CIFAR100",
    #         apply_data_normalization=True,
    #         apply_data_augmentation=True,
    #         add_corrupted_test=True,
    #         valid_size=0.95,
    #     )
    #     self.get_parts(dataset_conf, model_conf, self.seed)
    #     trainer_conf = trainer.TrainerConfig(
    #         comment="CIFAR100 Training Test",
    #         max_iter=2,
    #         verbose=False,
    #         add_noise=False,
    #         noise_snr=None,
    #         noise_std=None,
    #         noise_test={"noise_snr": [], "noise_std": [],},
    #         restore_best=False,
    #         lr_milestones=(1,),
    #         adaptive_lr=False,
    #         patience=1000,
    #     )
    #     score = self.run_training(trainer_conf)
    #     self.assertAlmostEqual(
    #         score, 11.1, places=1
    #     )
    #
    # def test_tiny_imagenet(self):
    #     print("===================================================", flush=True)
    #     print("==============TEST Tiny-ImageNet Training==========", flush=True)
    #     model_conf = model.ClassificationModelConfig(
    #         comment="TinyImageNet", dataset_cls="TinyImageNet", type="18",
    #     )
    #     dataset_conf = dataset.ImageDatasetConfig(
    #         comment="Minimal TinyImageNet",
    #         dataset_cls="TinyImageNet",
    #         apply_data_normalization=True,
    #         apply_data_augmentation=True,
    #         add_corrupted_test=True,
    #         valid_size=0.95,
    #     )
    #     self.get_parts(dataset_conf, model_conf, self.seed)
    #     trainer_conf = trainer.TrainerConfig(
    #         comment="TinyImageNet Training Test",
    #         max_iter=2,
    #         verbose=False,
    #         add_noise=False,
    #         noise_snr=None,
    #         noise_std=None,
    #         noise_test={"noise_snr": [], "noise_std": [],},
    #         restore_best=False,
    #         lr_milestones=(1,),
    #         adaptive_lr=False,
    #         patience=1000,
    #     )
    #     score = self.run_training(trainer_conf)
    #     self.assertAlmostEqual(
    #         score, 5.76, places=1
    #     )

    def test_imagenet(self):
        print("===================================================", flush=True)
        print("=================TEST ImageNet Training============", flush=True)
        model_conf = model.ClassificationModelConfig(
            comment="ImageNet", dataset_cls="ImageNet", type="18",
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
            patience=1000,
        )
        score = self.run_training(trainer_conf)
        self.assertAlmostEqual(
            score, 0.530752, places=1
        )

    def test_imagenet_pretrained(self):
        print("===================================================", flush=True)
        print("================TEST ImageNet Pretrained===========", flush=True)
        model_conf = model.ClassificationModelConfig(
            comment="ImageNet", dataset_cls="ImageNet", type="50", pretrained=True
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
            patience=1000,
        )
        score = self.run_training(trainer_conf)
        self.assertAlmostEqual(
            score, 76.1, places=1
        )

if __name__ == "__main__":
    unittest.main()
