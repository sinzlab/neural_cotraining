import unittest
import torch

from bias_transfer.configs import trainer
from bias_transfer.models import get_model_parameters
from bias_transfer.tests.main_loop_module import MainLoopModuleTest
from bias_transfer.trainer.main_loop_modules import LotteryTicketPruning


class LotteryTicketPruningTest(MainLoopModuleTest):
    def pre_epoch_test(self, model, epoch):
        mask_sum = sum([torch.sum(m).cpu().detach().item() for m in self.module.mask])
        p = (1-(self.percent/100)) ** (1 / self.rounds)
        self.assertAlmostEqual(mask_sum, self.total_parameters * p, places=-1)  # remaining parameters

    def post_backward_test(self, model):
        pass

    def test_module(self):
        self.rounds = 2
        self.percent = 80
        trainer_conf = trainer.TrainerConfig(
            comment="Minimal Training Test",
            max_iter=3,
            verbose=False,
            noise_test={"noise_snr": [], "noise_std": [],},
            restore_best=False,
            lr_milestones=(1, 2),
            adaptive_lr=False,
            patience=1000,
            lottery_ticket={
                "rounds": self.rounds,
                "percent_to_prune": self.percent,
                "pruning": True,
                "reinit": False,
                "global_pruning": True,
            },
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(device)
        self.module = LotteryTicketPruning(
            self.model, trainer_conf, "cuda", self.data_loaders["train"], self.seed
        )
        mask_sum = sum([torch.sum(m) for m in self.module.mask])  # should all be one
        self.total_parameters = 0
        for name, param in self.model.named_parameters():
            if "weight" in name:
                size = 1
                for l in list(param.size()):
                    size *= l
                self.total_parameters += size
        self.assertEqual(mask_sum, self.total_parameters)
        self.main_loop(
            self.model, self.data_loaders["train"], self.module, trainer_conf, device, epoch=0
        )


if __name__ == "__main__":
    unittest.main()
