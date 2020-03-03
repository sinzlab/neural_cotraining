class MainLoopModule(object):
    def __init__(self, config, device, data_loader, seed):
        self.config = config
        self.device = device
        self.seed = seed
        self.criterion = None

    def pre_epoch(self, model, train_mode):
        pass

    def pre_forward(self, model, inputs, shared_memory, train_mode):
        return model, inputs

    def post_forward(self, outputs, loss, extra_losses, applied_std=None, **kwargs):
        return outputs, loss
