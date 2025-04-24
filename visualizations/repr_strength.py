import logging
import os
import re

import matplotlib.pyplot as plt

import numpy as np

from visualizations.utils import calculate_representation_strength


def main():
    # plt.figure().set_figwidth(88)
    fig, ax = plt.subplots()
    num_tasks = 10
    results = {}
    dataset_name = "/home/greg/WYNIKI2/motivation-final/"
    methods = ['none', 'feature', 'logit', 'projected']
    colors = ["tab:orange", "tab:blue", "tab:green", "tab:purple", "k"]
    for i, method in enumerate(methods):
        log_name = dataset_name + method
        files = os.listdir(log_name)
        file = [i for i in files if "stdout" in i][0]
        with open(log_name + "/" + file) as f:
            log = f.readlines()
        # Remove everything from the log beside last task
        upper_bounds = [i for i, x in enumerate(log) if x == "### 0.95 of eigenvalues sum is explained by: ###\n"]
        log = log[upper_bounds[-1]:]
        results = calculate_representation_strength(log[1:11], num_tasks)

        plt.plot([_+1 for _ in range(num_tasks)], results, 'o-', color=colors[i], label=method, linewidth=3, markersize=10)

    plt.xticks([_+1 for _ in range(num_tasks)], fontsize=29)
    plt.yticks(fontsize=29)
    plt.ylim(0, 40)


    plt.xlabel("Task", fontsize=29)
    plt.ylabel("Eigenvalues explaining 95% of variance", fontsize=29)

    fig = plt.gcf()
    fig.set_size_inches(12, 10)

    # Shrink current axis's height by 10% on the bottom
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0 + box.height * 0.1,
    #                  box.width, box.height * 0.9])

    # Put a legend below current axis
    legend = ax.legend(loc='upper center', title="Distillation", fancybox=True, shadow=True, ncol=2, fontsize=29)
    plt.setp(legend.get_title(), fontsize=30)
    plt.savefig("representation-strength.png", dpi=200, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
