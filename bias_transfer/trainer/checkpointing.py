import os

import numpy as np
import pickle as pkl

import torch

from mlutils.training import copy_state


class Checkpointing:
    def __init__(
        self,
        model,
        scheduler,
        tracker,
        chkpt_options,
        maximize_score,
        call_back=None,
        hash=None,
    ):
        self.call_back = call_back
        self.hash = hash
        self.model = model
        self.scheduler = scheduler
        self.tracker = tracker
        self.chkpt_options = chkpt_options
        self.maximize_score = maximize_score

    def save(self, epoch, score, patience_counter):
        raise NotImplementedError

    def restore(self, restore_only_state=False):
        raise NotImplementedError


class RemoteCheckpointing(Checkpointing):
    def save(self, epoch, score, patience_counter):
        state = {
            "action": "save",
            "score": score,
            "maximize_score": self.maximize_score,
            "tracker": self.tracker.state_dict(),
            "patience_counter": patience_counter,
            **self.chkpt_options,
        }
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()
        self.call_back(
            epoch=epoch,
            model=self.model,
            state=state,
        )  # save model

    def restore(self, restore_only_state=False, action="last"):
        loaded_state = {
            "maximize_score": self.maximize_score,
            "action": action,
        }
        if not restore_only_state:
            loaded_state["tracker"] = self.tracker
            if self.scheduler is not None:
                loaded_state["scheduler"] = self.scheduler
        self.call_back(
            epoch=-1, model=self.model, state=loaded_state
        )  # load the last epoch if existing
        epoch = loaded_state.get("epoch", 0)
        patience_counter = loaded_state.get("patience_counter", -1)
        return epoch, patience_counter


class LocalCheckpointing(Checkpointing):
    def save(self, epoch, score, patience_counter):
        state = {
            "score": score,
            "epoch": epoch,
            "tracker": self.tracker,
            "patience_counter": patience_counter,
            "model": self.model.state_dict(),
        }
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()
        torch.save(state, f"./{self.hash}_{epoch}_{score}_chkpt.pth.tar")

        # select checkpoints to be kept
        checkpoints = [f for f in os.listdir("./") if self.hash in f]
        keep_checkpoints = []
        last_checkpoints = sorted(
            checkpoints, key=lambda chkpt: int(chkpt.split("_")[1]), reverse=True
        )
        keep_checkpoints += last_checkpoints[
            : self.chkpt_options.get("keep_last_n", 1)
        ]  # w.r.t. temporal order
        best_checkpoints = sorted(
            checkpoints,
            key=lambda chkpt: float(chkpt.split("_")[2]),
            reverse=self.maximize_score,
        )
        keep_checkpoints += best_checkpoints[
            : self.chkpt_options.get("keep_best_n", 1)
        ]  # w.r.t. performance
        # delete the others
        for chkpt in checkpoints:
            if not chkpt in keep_checkpoints:
                os.remove(chkpt)

    def restore(self, restore_only_state=False, action="last"):
        # list checkpoints:
        checkpoints = [f for f in os.listdir("./") if self.hash in f]
        if not checkpoints:
            return 0, -1
        if action == "last":
            chkpt = sorted(
                checkpoints, key=lambda chkpt: int(chkpt.split("_")[1]), reverse=True
            )[0]
        elif action == "best":
            chkpt = sorted(
                checkpoints,
                key=lambda chkpt: float(chkpt.split("_")[2]),
                reverse=self.maximize_score,
            )[0]

        state = torch.load(chkpt)
        if not restore_only_state:
            for elem in self.tracker:
                self.tracker.remove(elem)
            self.tracker += state["tracker"]
            if self.scheduler is not None:
                self.scheduler.load_state_dict(state["scheduler"])
        epoch = state.get("epoch", 0)
        patience_counter = state.get("patience_counter", -1)
        self.model.load_state_dict(state["model"])
        return epoch, patience_counter


class TemporaryCheckpointing(Checkpointing):
    def __init__(
        self, model, scheduler, tracker, chkpt_options, maximize_score, call_back=None
    ):
        super().__init__(
            model, scheduler, tracker, chkpt_options, maximize_score, call_back
        )
        # prepare state save
        self.score_save = np.infty * -1 if maximize_score else np.infty
        self.epoch_save = -1
        self.patience_counter_save = -1
        self.model_save = copy_state(model)
        self.scheduler_save = copy_state(scheduler) if scheduler is not None else {}
        self.tracker_save = copy_state(tracker)

    def save(self, epoch, score, patience_counter):
        self.score_save = score
        self.epoch_save = epoch
        self.patience_counter_save = patience_counter
        self.model_save = copy_state(self.model)
        self.scheduler_save = copy_state(self.scheduler) if self.scheduler else {}
        self.tracker_save = copy_state(self.tracker)

    def restore(self, restore_only_state=False):
        if self.epoch_save > -1:
            self.model.load_state_dict(self.model_save)
            if not restore_only_state:
                if self.scheduler:
                    self.scheduler.load_state_dict(self.scheduler_save)
                self.tracker.load_state_dict(self.tracker_save)
        return self.epoch_save, self.patience_counter_save
