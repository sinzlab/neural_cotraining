# import unittest
# from bias_transfer.configs import model, dataset, trainer
# from .test_minimal_training import MinimalTrainingTest
#
#
# class LotteryTicketPruningTest(MinimalTrainingTest):
#     def test_training_fixed_lr_schedule(self):
#         trainer_conf = trainer.TrainerConfig(
#             comment="Minimal Training Test",
#             max_iter=3,
#             verbose=False,
#             noise_test={"noise_snr": [], "noise_std": [],},
#             restore_best=False,
#             lr_milestones=(1, 2),
#             adaptive_lr=False,
#             patience=1000,
#         )
#         score = self.run_training(trainer_conf)
#         self.assertAlmostEqual(score, self.score)
#
#
# if __name__ == "__main__":
#     unittest.main()
