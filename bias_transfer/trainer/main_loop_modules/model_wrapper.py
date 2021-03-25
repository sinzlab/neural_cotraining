from functools import partial
from .main_loop_module import MainLoopModule


class ModelWrapper(MainLoopModule):
    def __init__(self, model, config, device, data_loader, seed):
        super().__init__(model, config, device, data_loader, seed)

    def pre_forward(self, model, inputs, shared_memory, train_mode, **kwargs):
        data_key = kwargs.pop("data_key", None)
        neural_set = kwargs.pop("neural_set", None)
        task_keys = kwargs.pop("task_keys", [])
        if self.mtl:
            if len(task_keys) == 1:
                if (data_key == "img_classification") or (task_keys[0] == "img_classification" and len(task_keys) == 1):
                    model_ = partial(model, neural_set=neural_set, data_key=data_key, classification=True)
                else:
                    model_ = partial(model, neural_set=neural_set, data_key=data_key)
            else:
                model_ = partial(model, neural_set=neural_set, data_key=data_key, classification=True, both=True)
        elif (data_key == "img_classification") or (task_keys[0] == "img_classification" and len(task_keys) == 1):
            model_ = model
        else:
            model_ = partial(model, data_key=data_key)
        return model_, inputs

    def post_forward(self, outputs, loss, targets, extra_losses, train_mode, **kwargs):
        task_keys = kwargs.pop("task_keys", [])
        if len(task_keys) == 1:
            outputs = {task_keys[0]: outputs}
            if self.return_classification_subset > 0 and task_keys[0] == "img_classification":
                outputs["img_classification"] = outputs["img_classification"][:,:self.return_classification_subset]
        else:
            outputs = {task: outputs[0] if task in ['v1', 'v4'] else outputs[1] for task in task_keys}
            if self.return_classification_subset > 0:
                outputs["img_classification"] = outputs["img_classification"][:,:self.return_classification_subset]
        return outputs, loss, targets