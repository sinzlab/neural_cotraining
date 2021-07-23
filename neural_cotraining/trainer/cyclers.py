from neuralpredictors.training.cyclers import cycle


class MTL_Cycler:
    def __init__(self, loaders, main_key="img_classification", ratio=1):
        self.main_key = (
            main_key  # data_key of the dataset whose batch ratio is always 1
        )
        if isinstance(loaders[main_key], dict):
            second_key = list(loaders[main_key].keys())[0]
            self.main_loader = loaders[main_key][second_key]
        else:
            self.main_loader = loaders[main_key]
        self.other_loaders = {k: loaders[k] for k in loaders.keys() if k != main_key}
        self.ratio = ratio  # number of neural batches vs. one batch from TIN
        self.num_batches = int(
            len(self.main_loader) * (ratio * len(self.other_loaders.keys()) + 1)
        )
        self.backward = False

    def generate_batch(self, main_cycle, other_cycles_dict):
        if self.ratio >= 1 or self.ratio == 0:
            for main_batch_idx in range(len(self.main_loader)):
                for neural_set in other_cycles_dict.keys():
                    for batch_idx in range(self.ratio):
                        key, loader = next(other_cycles_dict[neural_set])
                        yield (neural_set, key), loader
                self.backward = True
                yield self.main_key, main_cycle
                self.backward = False
        else:
            for i in range(len(self.main_loader)):
                for neural_set in other_cycles_dict.keys():
                    if (i + 1) % (1 / self.ratio) == 0:
                        key, loader = next(other_cycles_dict[neural_set])
                        yield (neural_set, key), loader
                self.backward = True
                yield self.main_key, main_cycle
                self.backward = False

    def __iter__(self):
        other_cycles_dict = {}
        for neural_set, loaders_set in self.other_loaders.items():
            other_cycles = {k: cycle(v) for k, v in loaders_set.items()}
            other_cycles_dict[neural_set] = cycle(other_cycles.items())
        main_cycle = cycle(self.main_loader)
        for k, loader in self.generate_batch(main_cycle, other_cycles_dict):
            yield k, next(loader)

    def __len__(self):
        return self.num_batches


class LongCycler:
    """
    Cycles through trainloaders until the loader with largest size is exhausted.
        Needed for dataloaders of unequal size (as in the monkey data).
    It can handle loaders in the framework of MTL trainer
    """

    def __init__(self, loaders, grad_accum_step=0):
        if isinstance(loaders[list(loaders.keys())[0]], dict):
            self.main_key = list(loaders.keys())[0]
            self.loaders = loaders[self.main_key]
        else:
            self.main_key = ""
            self.loaders = loaders
        self.max_batches = max([len(loader) for loader in self.loaders.values()])
        self.backward = False
        self.grad_accum_step = (
            len(self.loaders) if not grad_accum_step else grad_accum_step
        )

    def __iter__(self):
        cycles = [cycle(loader) for loader in self.loaders.values()]
        for k, loader, batch_idx in zip(
            cycle(self.loaders.keys()),
            (cycle(cycles)),
            range(len(self.loaders) * self.max_batches),
        ):
            if (batch_idx + 1) % self.grad_accum_step == 0:
                self.backward = True
            if self.main_key:
                yield (self.main_key, k), next(loader)
            else:
                yield k, next(loader)
            self.backward = False

    def __len__(self):
        return len(self.loaders) * self.max_batches


class ShortCycler:
    """
    Cycles through trainloaders until the loader with smallest size is exhausted.
        Needed for dataloaders of unequal size (as in the monkey data).
    """

    def __init__(self, loaders, grad_accum_step=0):
        if isinstance(loaders[list(loaders.keys())[0]], dict):
            self.main_key = list(loaders.keys())[0]
            self.loaders = loaders[self.main_key]
        else:
            self.main_key = ""
            self.loaders = loaders
        self.min_batches = min([len(loader) for loader in self.loaders.values()])
        self.backward = False
        self.grad_accum_step = (
            len(self.loaders) if not grad_accum_step else grad_accum_step
        )

    def __iter__(self):
        cycles = [cycle(loader) for loader in self.loaders.values()]
        for k, loader, batch_idx in zip(
            cycle(self.loaders.keys()),
            (cycle(cycles)),
            range(len(self.loaders) * self.min_batches),
        ):
            if (batch_idx + 1) % self.grad_accum_step == 0:
                self.backward = True
            if self.main_key:
                yield (self.main_key, k), next(loader)
            else:
                yield k, next(loader)
            self.backward = False

    def __len__(self):
        return len(self.loaders) * self.min_batches
