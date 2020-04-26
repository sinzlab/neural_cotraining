import math
import torch
import torch.backends.cudnn as cudnn
from bias_transfer.utils.io import load_checkpoint
import numpy as np
from matplotlib import cm
from sklearn.cluster import AgglomerativeClustering
from nnfabrik.main import *

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Analyzer:
    def __init__(
        self, keys_to_fetch=("comment",),
    ):
        self.keys_to_fetch = keys_to_fetch
        self.df = pd.DataFrame()

    def add_data(self, configs, table, transfer_level=0):
        self.df = self.df.append(self._load_data(configs, table, transfer_level))

    def _load_data(self, configs, table, transfer_level):
        # Select data:
        fetched = []
        for description, config in configs.items():
            if transfer_level < len(config.get_restrictions()):
                restricted = table & config.get_restrictions()[transfer_level]
            else:
                restricted = None
            if restricted:  # could be empty if entry is not computed yet
                fetch_res = restricted.fetch1("output", *self.keys_to_fetch)
                fetched.append((description.name,) + fetch_res)

        # Reformat to display nicely/ access easily:
        df = pd.DataFrame(fetched)
        if not fetched:
            return df
        df = df.rename(columns={0: "name", 2: "comment"})
        df = pd.concat([df.drop([1], axis=1), df[1].apply(pd.Series)], axis=1)
        df = df.rename(columns={0: "training_progress"})
        df = pd.concat([df.drop([1], axis=1), df[1].apply(pd.Series)], axis=1)
        return df

    def plot(
        self,
        to_plot="dev_noise_eval",
        noise_measure="std",
        save="",
        perf_measure="dev_eval",
        style="lighttalk",
    ):
        if not to_plot in ("c_test_eval", "c_test_loss"):
            fs = (16, 10) if "talk" in style else (12, 7.5)
            dpi = 200 if "talk" in style else 200
            sns.set()
            if "light" in style:
                sns.set_style("whitegrid")
            if "ticks" in style:
                sns.set_style("ticks")
            if "dark" in style:
                plt.style.use("dark_background")
            if "talk" in style:
                sns.set_context("talk")
            else:
                sns.set_context("paper")
            fig, ax = plt.subplots(figsize=fs, dpi=dpi)
        # Plot
        if to_plot in ("test_eval", "test_loss"):
            sns.barplot(x="name", y=to_plot, hue="name", data=self.df, ax=ax)
        elif to_plot in ("dev_noise_eval", "dev_noise_loss"):
            data = self.df[to_plot].apply(pd.Series)
            data = data["noise_" + noise_measure].apply(pd.Series)
            data = pd.concat([self.df["name"], data], axis=1)
            data.index = data.name
            del data["name"]
            data = data.stack().reset_index()
            data.columns = [
                "Training",
                "Noise in Evaluation (Standard deviation)",
                "Accuracy",
            ]
            data["Noise in Evaluation (Standard deviation)"] = data[
                "Noise in Evaluation (Standard deviation)"
            ].map(lambda x: float(x.split("_")[0]))
            sns.barplot(
                data=data,
                x="Noise in Evaluation (Standard deviation)",
                y="Accuracy",
                hue="Training",
                ax=ax,
            )
        elif to_plot in ("c_test_eval", "c_test_loss"):
            for group in (
                (
                    "shot_noise",
                    "impulse_noise",
                    "speckle_noise",
                    "gaussian_noise",
                    "defocus_blur",
                    "gaussian_blur",
                    "motion_blur",
                    "glass_blur",
                    "zoom_blur",
                    "brightness",
                    "fog",
                    "frost",
                    "snow",
                    "contrast",
                    "elastic_transform",
                    "pixelate",
                    "jpeg_compression",
                    "saturate",
                    "spatter",
                ),
            ):
                data = self.df[to_plot].apply(pd.Series)
                data_to_plot = pd.DataFrame()
                for corruption in group:
                    if corruption not in data.columns:
                        continue
                    data_ = data[corruption].apply(pd.Series)
                    data_ = pd.concat([self.df["name"], data_], axis=1)
                    data_["Corruption"] = corruption
                    data_to_plot = pd.concat([data_to_plot, data_], axis=0, sort=True)
                    data_to_plot.index = data_to_plot.name
                    del data_to_plot["name"]
                g = sns.FacetGrid(
                    data=data_to_plot,
                    col="Corruption",
                    col_wrap=4,
                    sharey=True,
                    sharex=True,
                    # height=4
                )

                def draw_heatmap(data, *args, **kwargs):
                    del data["Corruption"]
                    # print(data)
                    sns.heatmap(data, annot=True, cbar=False)
                g.map_dataframe(draw_heatmap)
                fig = g.fig
        elif to_plot in ("training_progress",):
            data = self.df[to_plot].apply(pd.Series)
            data = data.applymap(lambda x: x.get(perf_measure) if isinstance(x,dict) else None)
            data = pd.concat([self.df["name"], data], axis=1)
            data.index = data.name
            del data["name"]
            data = data.stack().reset_index()
            data.columns = ["name", "epoch", "score"]
            sns.lineplot(x="epoch", y="score", hue="name", data=data, ax=ax)
        else:
            print("Unknown plot option!")

        sns.despine(offset=10, trim=True)
        if to_plot in ("c_test_eval", "c_test_loss"):
            # remove ticks again (see: https://stackoverflow.com/questions/37860163/seaborn-despine-brings-back-the-ytick-labels)
            # loop over the non-left axes:
            for i, ax in enumerate(g.axes.flat):
                if i % 4 != 0:
                    # get the yticklabels from the axis and set visibility to False
                    for label in ax.get_yticklabels():
                        label.set_visible(False)
                    ax.yaxis.offsetText.set_visible(False)
                if i < len(g.axes) - 4:
                    # get the xticklabels from the axis and set visibility to False
                    for label in ax.get_xticklabels():
                        label.set_visible(False)
                    ax.xaxis.offsetText.set_visible(False)
        if "talk" in style:
            plt.legend(fontsize=14, title_fontsize="14")
        if save:
            fig.savefig(
                save + "_" + style,
                facecolor=fig.get_facecolor(),
                edgecolor=fig.get_edgecolor(),
                bbox_inches="tight",
            )


def print_table_for_excel(table):
    prior_columns = 1
    keys = []
    for key in table.fetch(as_dict=True)[0].keys():
        if "comment" in key:
            keys.append(key)
        if key == "transfer_trainer_hash" or key == "transfer_trainer_config":
            keys.append(key)
            prior_columns = 3
    # heading
    row = table.fetch("output", as_dict=True)[0]["output"][1]["dev_noise_acc"]
    print("," * prior_columns, end="")
    for key in row.keys():
        print(key + ("," * (len(row[key]))), end="")
    print()
    print("," * prior_columns, end="")
    for key in row.keys():
        for sub_key in row[key].keys():
            print(sub_key.split("_")[0], end=",")
    print()
    # content
    for row in table.fetch("output", *keys, as_dict=True):
        comment = []
        extra = []
        for k, v in row.items():
            if k == "output" or "hash" in k:
                continue
            elif "config" in k:
                extra = [
                    "freeze_{}".format(v["freeze"]),
                    "reset_{}".format(v["reset_linear"]),
                ]
            else:
                comment.append(v)
        print(".".join(comment), end=", ")
        if extra:
            print(",".join(extra), end=", ")
        output = row["output"]
        final_results = output[1]["dev_noise_acc"]
        for key in final_results.keys():
            for res in final_results[key].items():
                print(res[1], end=",")
        print()

