class MainLoopModule(object):
    def __init__(self, model, config, device, data_loader, seed):
        self.config = config
        self.device = device
        self.seed = seed
        self.criterion = None
        self.data_loader_keys_nr = len(config.loss_functions)

    def pre_epoch(self, model, train_mode, epoch):
        pass

    def pre_forward(self, model, inputs, shared_memory, train_mode, **kwargs):
        return model, inputs

    def post_forward(self, outputs, loss, targets, extra_losses, train_mode, **kwargs):
        return outputs, loss, targets

    def post_backward(self, model, **kwargs):
        pass
