import numpy as np
import torch
from torch import optim, nn
from torch.backends import cudnn as cudnn
from tqdm import tqdm

from bias_transfer.tests._base import BaseTest
from bias_transfer.trainer.main_loop import move_data


class MainLoopModuleTest(BaseTest):
    def pre_epoch_test(self, model, epoch):
        pass

    def pre_forward_test(self, model, inputs, shared_memory):
        pass

    def post_forward_test(self, outputs, loss, targets, module_losses, **kwargs):
        pass

    def post_backward_test(self, model):
        pass

    def main_loop(
        self, model, data_loader, module, config, device, epoch: int = 0
    ):
        optimizer = getattr(optim, config.optimizer)(
            model.parameters(), **config.optimizer_options
        )
        n_iterations = len(data_loader)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if device == "cuda":
            cudnn.benchmark = False
            cudnn.deterministic = True
            torch.cuda.manual_seed(self.seed)
        criterion = getattr(nn, config.loss_function)()
        model.train()
        epoch_loss, correct, total, module_losses, collected_outputs = 0, 0, 0, {}, []
        if hasattr(
            tqdm, "_instances"
        ):  # To have tqdm output without line-breaks between steps
            tqdm._instances.clear()
        with torch.enable_grad():
            with tqdm(
                enumerate(data_loader),
                total=n_iterations,
                desc="{} Epoch {}".format("Train", epoch),
            ) as t:

                module.pre_epoch(model, True, epoch)
                self.pre_epoch_test(model, epoch)

                optimizer.zero_grad()

                for batch_idx, batch_data in t:
                    # Pre-Forward
                    loss = torch.zeros(1, device=device)
                    inputs, targets, data_key, batch_dict = move_data(
                        batch_data, device, False
                    )
                    shared_memory = {}  # e.g. to remember where which noise was applied
                    model, inputs = module.pre_forward(
                        model, inputs, shared_memory, True
                    )
                    self.pre_forward_test(model, inputs, shared_memory)
                    # Forward
                    outputs = model(inputs)
                    # Post-Forward
                    outputs, loss, targets = module.post_forward(
                        outputs, loss, targets, module_losses, True, **shared_memory
                    )
                    self.post_forward_test(
                        outputs, loss, targets, module_losses, **shared_memory
                    )
                    loss += criterion(outputs, targets)
                    epoch_loss += loss.item()

                    # Book-keeping
                    def average_loss(loss_):
                        return loss_ / (batch_idx + 1)

                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    eval = 100.0 * correct / total

                    t.set_postfix(
                        eval=eval,
                        loss=average_loss(epoch_loss),
                        **{k: average_loss(l) for k, l in module_losses.items()}
                    )

                    # Backward
                    loss.backward()
                    module.post_backward(model)
                    self.post_backward_test(model)

                    optimizer.step()
                    optimizer.zero_grad()

        return (
            eval,
            average_loss(epoch_loss),
            {k: average_loss(l) for k, l in module_losses.items()},
        )
