import logging
import os
import re

import matplotlib.pyplot as plt

import numpy as np
import torch

from visualizations.utils import extract_det


def main():
    # plt.figure().set_figwidth(88)
    fig, ax = plt.subplots()
    num_tasks = 10
    results = {}
    dataset_name = "/home/greg/WYNIKI2/cifar100-determinant/"
    methods = ['0', '0.01']
    colors = ["tab:blue", "tab:orange"]
    for i, method in enumerate(methods):
        log_name = dataset_name + "alpha=" + method
        files = os.listdir(log_name)
        file = [i for i in files if "stdout" in i][0]
        with open(log_name + "/" + file) as f:
            log = f.readlines()
        # Remove everything from the log beside last task
        upper_bounds = [i for i, x in enumerate(log) if "### Training backbone ###" in x]
        log = log[upper_bounds[0]:]
        results = extract_det(log[4:204])
        # for task in range(10):
        #     upper_bounds = [i for i, x in enumerate(log) if "Rank for class 0:" in x]
        #     lines = log[upper_bounds[task]:]
        #     results.extend(calculate_ranks(lines[10*task:10*(task+1)]))
        results = torch.tensor(results)
        results = torch.nn.functional.avg_pool1d(results.unsqueeze(0), 5, stride=1, padding=2).squeeze(0)

        plt.plot([_+1 for _ in range(200)], results, 'o-', color=colors[i], label=method, linewidth=3, markersize=10)

    plt.xticks([40*_+1 for _ in range(5)], fontsize=34)
    plt.yticks(fontsize=34)
    plt.ylim(0, 0.01)
    plt.title("Determinant in task 1", fontsize=44)

    plt.xlabel("Epoch", fontsize=40)
    plt.ylabel("Determinant", fontsize=40)

    fig = plt.gcf()
    fig.set_size_inches(16.5, 10.5)

    # Shrink current axis's height by 10% on the bottom
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0 + box.height * 0.1,
    #                  box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper right', fancybox=True, shadow=True, ncol=2, fontsize=28)

    plt.savefig("dets.png", dpi=200, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
