from neural_cotraining.trainer.main_loop_modules.main_loop_module import MainLoopModule


class OutputSelector(MainLoopModule):
    def __init__(self, model, config, device, data_loader, seed):
        super().__init__(model, config, device, data_loader, seed)

    def post_forward(self, outputs, loss, targets, extra_losses, train_mode, **kwargs):
        return outputs[1], loss, targets
