import logging
import os
import re

import matplotlib.pyplot as plt

import numpy as np
import torch

from visualizations.utils import extract_det, extract_losses


def main():

    fig, ax = plt.subplots()
    dataset_name = "/home/greg/WYNIKI2/v3/"
    methods = ['beta=1_alpha=1']

    for i, method in enumerate(methods):
        log_name = dataset_name + method
        files = os.listdir(log_name)
        file = [i for i in files if "stdout" in i][0]
        with open(log_name + "/" + file) as f:
            log = f.readlines()
        # Remove everything from the log beside last task
        indices = [i for i, x in enumerate(log) if "Task  1\n" in x]
        log = log[indices[0]:][6:206]
        ce, ac, kd = extract_losses(log)
        # for task in range(10):
        #     upper_bounds = [i for i, x in enumerate(log) if "Rank for class 0:" in x]
        #     lines = log[upper_bounds[task]:]
        #     results.extend(calculate_ranks(lines[10*task:10*(task+1)]))

    plt.plot([_+1 for _ in range(200)], ce, 'o-', color="tab:orange", label="CE", linewidth=3, markersize=10)
    plt.plot([_+1 for _ in range(200)], ac, 'o-', color="tab:blue", label="AC", linewidth=3, markersize=10)
    plt.plot([_+1 for _ in range(200)], kd, 'o-', color="tab:green", label="KD", linewidth=3, markersize=10)

    plt.xticks([40*_+1 for _ in range(5)], fontsize=34)
    plt.yticks(fontsize=34)
    plt.ylim(-1, 2)
    plt.title("Losses in second task", fontsize=44)

    plt.xlabel("Epoch", fontsize=40)
    plt.ylabel("Value", fontsize=40)

    fig = plt.gcf()
    fig.set_size_inches(16.5, 10.5)

    # Shrink current axis's height by 10% on the bottom
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0 + box.height * 0.1,
    #                  box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper right', fancybox=True, shadow=True, ncol=2, fontsize=28)

    plt.savefig("losses.png", dpi=200, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
