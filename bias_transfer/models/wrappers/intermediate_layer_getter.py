import functools
from collections import OrderedDict
from torch import nn


# using wonder's beautiful simplification:
# https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


class IntermediateLayerGetter(nn.Module):
    def __init__(self, model, return_layers, keep_output=True):
        """Wraps a Pytorch module to get intermediate values
        Original implementation by https://github.com/sebamenabar/Pytorch-IntermediateLayerGetter

        This version is modified to act feed through all actions to the underlying model
        (including accessing attributes and moving to GPU)

        Arguments:
            model {nn.module} -- The Pytorch module to call
            return_layers {dict} -- Dictionary with the selected submodules
            to return the output (format: {[current_module_name]: [desired_output_name]},
            current_module_name can be a nested submodule, e.g. submodule1.submodule2.submodule3)

        Keyword Arguments:
            keep_output {bool} -- If True model_output contains the final model's output
            in the other case model_output is None (default: {True})

        Returns:
            (mid_outputs {OrderedDict}, model_output {any}) -- mid_outputs keys are
            your desired_output_name (s) and their values are the returned tensors
            of those submodules (OrderedDict([(desired_output_name,tensor(...)), ...).
            See keep_output argument for model_output description.
            In case a submodule is called more than one time, all it's outputs are
            stored in a list.
        """
        super().__init__()
        self._model = model
        self.return_layers = return_layers
        self.keep_output = keep_output

    def __getattribute__(self, name):
        if name == "_model":
            return object.__getattribute__(self, name)
        else:
            try:
                return object.__getattribute__(self, name)
            except AttributeError:
                return getattr(self._model, name)

    def __call__(self, *args, **kwargs):
        ret = OrderedDict()
        handles = []
        for name, new_name in self.return_layers.items():
            layer = rgetattr(self._model, name)

            def hook(module, input, output, new_name=new_name):
                if new_name in ret:
                    if type(ret[new_name]) is list:
                        ret[new_name].append(output)
                    else:
                        ret[new_name] = [ret[new_name], output]
                else:
                    ret[new_name] = output

            try:
                h = layer.register_forward_hook(hook)
            except AttributeError as e:
                raise AttributeError(f"Module {name} not found")
            handles.append(h)

        if self.keep_output:
            output = self._model(*args, **kwargs)
        else:
            self._model(*args, **kwargs)
            output = None

        for h in handles:
            h.remove()

        return ret, output
