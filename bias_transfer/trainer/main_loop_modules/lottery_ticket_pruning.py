import numpy as np
import torch
import copy

from bias_transfer.utils import weight_reset
from .main_loop_module import MainLoopModule

EPS = 1e-6


class LotteryTicketPruning(MainLoopModule):
    """
    Based on the implementation from https://github.com/rahulvigneswaran/Lottery-Ticket-Hypothesis-in-Pytorch
    (therefore indirectly from https://github.com/ktkth5/lottery-ticket-hyopothesis)
    """

    def __init__(self, model, config, device, data_loader, seed):
        super().__init__(model, config, device, data_loader, seed)
        if self.config.lottery_ticket.get("pruning", True):
            n_epochs = self.config.max_iter
            n_rounds = self.config.lottery_ticket.get("rounds", 1)
            percent_to_prune = self.config.lottery_ticket.get("percent_to_prune", 80)
            self.percent_per_round = (
                1 - (1 - (percent_to_prune / 100)) ** (1 / n_rounds)
            ) * 100
            self.reset_epochs = list(range(0, n_epochs, n_epochs // n_rounds))
            print("percent per round:", self.percent_per_round)
            print("reset epochs:", list(self.reset_epochs))

            # create initial (empty mask):
            self.mask = self.make_empty_mask(model)

            # save initial state_dict to reset to this point later:
            if not self.config.lottery_ticket.get("reinit"):
                self.initial_state_dict = copy.deepcopy(model.state_dict())

    def pre_epoch(self, model, train_mode, epoch):
        if (
            self.config.lottery_ticket.get("pruning", True)
            and epoch in self.reset_epochs
            and epoch > 0  # validation calls this with epoch = 0
        ):
            # Prune the network, i.e. update the mask
            self.prune_by_percentile(model, self.percent_per_round)
            self.reset_initialization(model, self.config.lottery_ticket.get("reinit"))

    def post_backward(self, model, **kwargs):
        # Freezing Pruned weights by making their gradients Zero
        for name, p in model.named_parameters():
            if "weight" in name:
                tensor = torch.abs(p.data)
                grad_tensor = p.grad.data
                p.grad.data = torch.where(
                    tensor < EPS, torch.zeros_like(grad_tensor), grad_tensor
                )

    def prune_by_percentile(self, model, percent):
        # Calculate percentile value
        step = 0
        if self.config.lottery_ticket.get("global_pruning"):
            alive_tensors = []
            for name, param in model.named_parameters():
                if "weight" in name:  # We do not prune bias term
                    tensor = param.data
                    alive_tensors.append(
                        tensor[torch.nonzero(tensor, as_tuple=True)]
                    )  # flattened array of nonzero values
            alive = torch.cat(alive_tensors)
            percentile_value = np.percentile(torch.abs(alive).cpu().numpy(), percent)

        for name, param in model.named_parameters():
            if "weight" in name:  # We do not prune bias term
                tensor = param.data
                device = tensor.device
                if not self.config.lottery_ticket.get("global_pruning"):
                    # print(nonzero)
                    alive = tensor[
                        torch.nonzero(tensor, as_tuple=True)
                    ]  # flattened array of nonzero values
                    abs_alive = torch.abs(alive).cpu().numpy()
                    percentile_value = np.percentile(abs_alive, percent)

                # Convert Tensors to numpy and calculate
                new_mask = torch.where(
                    torch.abs(tensor) < torch.tensor(percentile_value, device=device),
                    torch.zeros_like(self.mask[step]),
                    self.mask[step],
                )

                # Apply new weight and mask
                param.data = tensor * new_mask
                self.mask[step] = new_mask
                step += 1

    def make_empty_mask(self, model):
        """
        Function to make an empty mask of the same size as the model
        :param model:
        :return: mask
        """
        step = 0
        for name, param in model.named_parameters():
            if "weight" in name:
                step = step + 1
        mask = [None] * step
        step = 0
        for name, param in model.named_parameters():
            if "weight" in name:
                tensor = param.data
                mask[step] = torch.ones_like(tensor, device=tensor.device)
                step = step + 1
        return mask

    def reset_initialization(self, model, reinit=False):
        print("Reset init.......")
        if reinit:
            model.apply(weight_reset)
        step = 0
        for name, param in model.named_parameters():
            init = param.data if reinit else self.initial_state_dict[name]
            if "weight" in name:
                param.data = self.mask[step] * init
                step = step + 1
            if "bias" in name:
                param.data = init
