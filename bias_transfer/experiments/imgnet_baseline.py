from . import Description
from bias_transfer.configs.experiment import Experiment
from bias_transfer.configs import model, dataset, trainer

experiments = {}

#lrs = {"Adam" : [0.0005], "RMSprob": [0.1, 0.01]}
for opt in ["Adam", "RMSprob"]:
    # for lr in lrs[opt]:
    #     for seed in (44,):
    #         trainer_name = "{}_lr_{}".format(opt, lr)
    #         experiments[Description(name="TinyImgnet_VGG19bn_" + trainer_name, seed=seed)] = Experiment(
    #                 dataset=dataset.TinyImageNet(description="TinyImageNet"),
    #                 model=model.TinyImageNet(description="TinyImageNet_VGG19"),
    #                 trainer=trainer.TrainerConfig(description=trainer_name, optimizer=opt,
    #                                               lr=lr, lr_decay=0.1, num_epochs=90,
    #                                               lr_milestones=(30,60),
    #                                               noise_test={}),
    #                 seed=seed)
    for lr in [0.1, 0.01]:
        for seed in (44,):
            if (opt != "RMSprob") or (lr != 0.01):
                trainer_name = "{}_lr_{}".format(opt, lr)
                experiments[Description(name="TinyImgnet_ResNet_" + trainer_name, seed=seed)] = Experiment(
                        dataset=dataset.TinyImageNet(description="TinyImageNet"),
                        model=model.TinyImageNet(description="TinyImageNet"),
                        trainer=trainer.TrainerConfig(description=trainer_name, optimizer=opt,
                                                      lr=lr, lr_decay=0.1, num_epochs=90,
                                                      lr_milestones=(30,60)),
                        seed=seed)
            # else:
            #     trainer_name = "{}_lr_{}".format(opt, lr)
            #     experiments[Description(name="TinyImgnet_ResNet", seed=seed)] = Experiment(
            #         dataset=dataset.TinyImageNet(description="TinyImageNet"),
            #         model=model.TinyImageNet(description="TinyImageNet"),
            #         trainer=trainer.TrainerConfig(description=trainer_name, optimizer=opt,
            #                                       lr=lr, lr_decay=0.1, num_epochs=90,
            #                                       lr_milestones=(30, 60)),
            #         seed=seed)









# #for the rest of resnets
# for opt in ["Adam", "RMSprob"]:
#     for lr in [0.1, 0.01]:
#         for seed in (44,):
#             if (opt != "RMSprob") or (lr != 0.01):
#                 trainer_name = "{}_lr_{}".format(opt, lr)
#                 experiments[Description(name="TinyImgnet_ResNet_" + trainer_name, seed=seed)] = Experiment(
#                         dataset=dataset.TinyImageNet(description="TinyImageNet"),
#                         model=model.TinyImageNet(description="TinyImageNet"),
#                         trainer=trainer.TrainerConfig(description=trainer_name, optimizer=opt,
#                                                       lr=lr, lr_decay=0.1, num_epochs=90,
#                                                       lr_milestones=(30,60)),
#                         seed=seed)
#             else:
#                 trainer_name = "{}_lr_{}".format(opt, lr)
#                 experiments[Description(name="TinyImgnet_ResNet", seed=seed)] = Experiment(
#                     dataset=dataset.TinyImageNet(description="TinyImageNet"),
#                     model=model.TinyImageNet(description="TinyImageNet"),
#                     trainer=trainer.TrainerConfig(description=trainer_name, optimizer=opt,
#                                                   lr=lr, lr_decay=0.1, num_epochs=90,
#                                                   lr_milestones=(30, 60)),
#                     seed=seed)


# "TinyImgnet_ResNet" was with rmsprop + 0.01 lr

# shahd_tinyimgnet_baseline_resnet = schema
# first resnet experiment _ second layer had strides of 2
# for seed in (8,):
#     # Clean baseline:
#     experiments[Description(name="Clean", seed=seed)] = Experiment(
#         dataset=dataset.TinyImageNet(description=""),
#         model=model.TinyImageNet(description=""),
#         trainer=trainer.TrainerConfig(description=""),
#         seed=seed)