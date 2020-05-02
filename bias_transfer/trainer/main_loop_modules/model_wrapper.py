from functools import partial
from .main_loop_module import MainLoopModule


class ModelWrapper(MainLoopModule):
    def __init__(self, model, config, device, data_loader, seed):
        super().__init__(model, config, device, data_loader, seed)

    def pre_forward(self, model, inputs, shared_memory, train_mode, **kwargs):
        data_key = kwargs.pop("data_key", None)
        if self.data_loader_keys_nr > 1:
            if data_key == "img_classification":
                model_ = partial(model, data_key=data_key, classification=True)
            else:
                model_ = partial(model, data_key=data_key)
        elif data_key == "img_classification":
            model_ = model
        else:
            model_ = partial(model, data_key=data_key)
        return model_, inputs
