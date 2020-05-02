import unittest
from bias_transfer.configs import trainer, model, dataset
from bias_transfer.tests._base import BaseTest


class ModelTest(BaseTest):
    def test_noise_adv_training(self):
        print("===================================================", flush=True)
        print("============TEST noise-adversarial training========", flush=True)
        model_conf = model.ClassificationModelConfig(
            comment="CIFAR10",
            dataset_cls="CIFAR10",
            type="18",
            noise_adv_regression=True,
        )
        self.get_parts(self.dataset_conf, model_conf, self.seed)
        trainer_conf = trainer.TrainerConfig(
            comment="Noise Adversarial Training Test",
            max_iter=2,
            verbose=False,
            add_noise=True,
            noise_snr=None,
            noise_std={0.08: 0.1, 0.12: 0.1, 0.18: 0.1, 0.26: 0.1, 0.38: 0.1, -1: 0.5,},
            noise_adv_regression=True,
            noise_test={"noise_snr": [], "noise_std": [],},
            restore_best=False,
            lr_milestones=(1,),
            adaptive_lr=False,
            patience=1000,
        )
        score = self.run_training(trainer_conf)
        self.assertAlmostEqual(score, 11.5, places=1)

    def test_representation_matching(self):
        print("===================================================", flush=True)
        print("=======TEST representation matching training ======", flush=True)
        model_conf = model.ClassificationModelConfig(
            comment="CIFAR10",
            dataset_cls="CIFAR10",
            type="18",
            representation_matching=True,
        )
        self.get_parts(self.dataset_conf, model_conf, self.seed)
        trainer_conf = trainer.TrainerConfig(
            comment="Representation Matching Training Test",
            max_iter=2,
            verbose=False,
            add_noise=True,
            noise_snr=None,
            noise_std={0.08: 0.1, 0.12: 0.1, 0.18: 0.1, 0.26: 0.1, 0.38: 0.1, -1: 0.5,},
            noise_test={"noise_snr": [], "noise_std": [],},
            restore_best=False,
            lr_milestones=(1,),
            adaptive_lr=False,
            patience=1000,
            representation_matching={
                "representation": "core",
                "criterion": "cosine",
                "second_noise_std": {(0, 1.0): 1.0},
                "lambda": 1.0,
            },
        )
        score = self.run_training(trainer_conf)
        self.assertAlmostEqual(score, 42.36, places=1)

    def test_resnet_50(self):
        print("===================================================", flush=True)
        print("================TEST ResNet50 Training=============", flush=True)
        model_conf = model.ClassificationModelConfig(
            comment="CIFAR10", dataset_cls="CIFAR10", type="50",
        )
        self.get_parts(self.dataset_conf, model_conf, self.seed)
        trainer_conf = trainer.TrainerConfig(
            comment="ResNet50 Training Test",
            max_iter=2,
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
            score, 9.44, places=1
        )  # ResNet50 seems too large to learn anything from so little data

    def test_vgg_19(self):
        print("===================================================", flush=True)
        print("==================TEST VGG19 Training==============", flush=True)
        model_conf = model.ClassificationModelConfig(
            comment="CIFAR10",
            dataset_cls="CIFAR10",
            type="vgg19_bn",
            cnn_builder="vgg",
        )
        self.get_parts(self.dataset_conf, model_conf, self.seed)
        trainer_conf = trainer.TrainerConfig(
            comment="VGG19 Training Test",
            max_iter=2,
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
        self.assertAlmostEqual(score, 15.2, places=1)


if __name__ == "__main__":
    unittest.main()
