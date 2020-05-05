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
            type="resnet18",
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
        self.assertAlmostEqual(score, 15.68, places=1)
        self.setUpClass()

    def test_noise_adv_training_vgg(self):
        print("===================================================", flush=True)
        print("========TEST noise-adversarial training (VGG)======", flush=True)
        model_conf = model.ClassificationModelConfig(
            comment="CIFAR10",
            dataset_cls="CIFAR10",
            type="vgg19_bn",
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
        self.assertAlmostEqual(score, 11.4, places=1)
        self.setUpClass()

    def test_representation_matching(self):
        print("===================================================", flush=True)
        print("=======TEST representation matching training ======", flush=True)
        model_conf = model.ClassificationModelConfig(
            comment="CIFAR10",
            dataset_cls="CIFAR10",
            type="resnet18",
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
        self.assertAlmostEqual(score, 37.96, places=1)
        self.setUpClass()

    def test_representation_matching_vgg(self):
        print("===================================================", flush=True)
        print("====TEST representation matching training (VGG)====", flush=True)
        model_conf = model.ClassificationModelConfig(
            comment="CIFAR10",
            dataset_cls="CIFAR10",
            type="vgg19_bn",
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
        self.assertAlmostEqual(score, 10.0, places=1)
        self.setUpClass()

    def test_resnet_50(self):
        print("===================================================", flush=True)
        print("================TEST ResNet50 Training=============", flush=True)
        model_conf = model.ClassificationModelConfig(
            comment="CIFAR10", dataset_cls="CIFAR10", type="resnet50",
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
            score, 9.56, places=1
        )  # ResNet50 seems too large to learn anything from so little data
        # reset:
        self.setUpClass()

    def test_vgg_19(self):
        print("===================================================", flush=True)
        print("==================TEST VGG19 Training==============", flush=True)
        model_conf = model.ClassificationModelConfig(
            comment="CIFAR10",
            dataset_cls="CIFAR10",
            type="vgg19_bn",
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
        self.assertAlmostEqual(score, 17.64, places=1)
        # reset:
        self.setUpClass()


if __name__ == "__main__":
    unittest.main()
