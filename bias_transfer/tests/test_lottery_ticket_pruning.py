import unittest
import torch

from bias_transfer.configs import trainer
from bias_transfer.tests._main_loop_module import MainLoopModuleTest
from bias_transfer.trainer.main_loop_modules import LotteryTicketPruning
from bias_transfer.models.utils import weight_reset


class LotteryTicketPruningTest(MainLoopModuleTest):
    def pre_epoch_test(self, model, epoch):
        mask_sum = sum([torch.sum(m).cpu().detach().item() for m in self.module.mask])
        p = (1 - (self.percent / 100)) ** (1 / self.rounds)
        self.assertAlmostEqual(
            mask_sum, self.total_parameters * p, places=-1
        )  # remaining parameters

    def post_backward_test(self, model):
        step = 0
        for name, p in model.named_parameters():
            if "weight" in name:
                grad_tensor = p.grad.data
                grad_masked = (grad_tensor != 0).int()
                self.assertTrue(
                    torch.all(torch.eq(grad_masked, self.module.mask[step]))
                    .cpu()
                    .detach()
                    .item()
                )
                step += 1

    def test_module(self):
        print("===================================================", flush=True)
        print("=====TEST the individual module components=========", flush=True)
        self.rounds = 1
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
                "round_length": 1,
                "percent_to_prune": self.percent,
                "pruning": True,
                "reinit": False,
                "global_pruning": True,
            },
        )
        self.model.apply(weight_reset)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.module = LotteryTicketPruning(
            self.model, trainer_conf, device, self.data_loaders["train"], self.seed
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
            self.model,
            self.data_loaders["train"],
            self.module,
            trainer_conf,
            device=device,
            epoch=2,
        )

    def test_training(self):
        print("===================================================", flush=True)
        print("===========TEST complete training==================", flush=True)
        rounds = 2
        round_length = 3
        percent = 69
        trainer_conf = trainer.TrainerConfig(
            comment="Minimal Training Test",
            max_iter=3,
            verbose=False,
            noise_test={"noise_snr": [], "noise_std": [],},
            restore_best=False,
            lr_milestones=None,
            adaptive_lr=False,
            patience=1000,
            lottery_ticket={
                "rounds": rounds,
                "round_length": round_length,
                "percent_to_prune": percent,
                "pruning": True,
                "reinit": False,
                "global_pruning": True,
            },
        )
        score = self.run_training(trainer_conf)
        zero_parameters, total_parameters = 0, 0
        for name, param in self.model.named_parameters():
            if "weight" in name:
                zero_parameters += torch.sum((param.data == 0).int()).item()
                size = 1
                for l in list(param.size()):
                    size *= l
                total_parameters += size
        self.assertAlmostEqual(
            zero_parameters, total_parameters * (percent / 100), places=-2
        )


if __name__ == "__main__":
    unittest.main()
