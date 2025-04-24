import logging
import os
import re

import matplotlib.pyplot as plt

import numpy as np

from visualizations.utils import calculate_ranks


def main():
    # plt.figure().set_figwidth(88)
    fig, ax = plt.subplots()
    num_tasks = 10
    results = {}
    dataset_name = "/home/greg/WYNIKI2/imagenet-10x10/"
    methods = ['mean', 'cov', 'kl']
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for i, method in enumerate(methods):
        log_name = dataset_name + method
        files = os.listdir(log_name)
        file = [i for i in files if "stdout" in i][0]
        with open(log_name + "/" + file) as f:
            log = f.readlines()
        # Remove everything from the log beside last task
        upper_bounds = [i for i, x in enumerate(log) if "Old val mean diff" in x]
        log = log[upper_bounds[-1]:]
        results = calculate_ranks(log[:100])
        # for task in range(10):
        #     upper_bounds = [i for i, x in enumerate(log) if "Rank for class 0:" in x]
        #     lines = log[upper_bounds[task]:]
        #     results.extend(calculate_ranks(lines[10*task:10*(task+1)]))
        results = [sum(results[10*i:10*(i+1)]) / 10 for i in range(10)]
        plt.plot([_+1 for _ in range(10)], results, 'o-', color=colors[i], label=method, linewidth=3, markersize=10)

    plt.xticks([_+1 for _ in range(10)], fontsize=34)
    plt.yticks(fontsize=34)
    plt.ylim(1, 90)
    plt.title("Covariance ranks per class", fontsize=44)

    plt.xlabel("Task", fontsize=40)
    plt.ylabel("Rank", fontsize=40)

    fig = plt.gcf()
    fig.set_size_inches(15, 10)

    # Shrink current axis's height by 10% on the bottom
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0 + box.height * 0.1,
    #                  box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper right', fancybox=True, shadow=True, ncol=2, fontsize=28)

    plt.savefig("ranks.png", dpi=200, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
