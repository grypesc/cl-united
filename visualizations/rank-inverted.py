import logging
import os
import re

import matplotlib.pyplot as plt

import numpy as np

from visualizations.utils import calculate_ranks, extract_mahalanobis


def main():

    fig, ax1 = plt.subplots()
    fig = plt.gcf()
    fig.set_size_inches(12, 10)

    mahalanobis_per_task = []

    path = "/home/greg/WYNIKI2/motivation-final/logit/"

    files = os.listdir(path)
    file = [i for i in files if "stdout" in i][0]
    with open(path + file) as f:
        log = f.readlines()

    upper_bounds = [i for i, x in enumerate(log) if "Rank for class 0:" in x]
    ranks = log[upper_bounds[-1]:]
    results = calculate_ranks(ranks[:100])

    rank_per_task = [sum(results[10*i:10*(i+1)]) / 10 for i in range(10)]

    upper_bounds = [i for i, x in enumerate(log) if "Inverted Covs:" in x]
    line = log[upper_bounds[-2]]
    inverted_per_task = extract_mahalanobis(line)

    err1 = [0.56, 0.52, 0.59, 0.61, 0.64, 0.64, 0.67, 0.68, 0.75, 0.85]
    ax1.plot([_+1 for _ in range(10)], rank_per_task, 'o-', color="black", label="Rank", linewidth=3, markersize=10)
    ax1.errorbar([_+1 for _ in range(10)], rank_per_task, yerr=err1, fmt='o-', color="black")
    ax1.set_ylabel('Covariance matrices rank', color="black", fontsize=28)
    ax1.tick_params(axis='y', labelcolor="black")
    plt.yticks(fontsize=28)
    plt.xlabel("Task", fontsize=34)
    plt.xticks([_+1 for _ in range(10)], fontsize=28)

    err2 = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    ax2 = ax1.twinx()
    ax2.plot([_+1 for _ in range(10)], inverted_per_task, 'o-', color="tab:green", label="Translation error (cm)", linewidth=3, markersize=10)
    ax2.errorbar([_+1 for _ in range(10)], inverted_per_task, yerr=err2, fmt='o-', color="tab:green")
    ax2.set_ylabel('Inverted covariance matrices norm', color="tab:green", fontsize=28)
    ax2.tick_params(axis='y', labelcolor="tab:green")

    plt.yticks(fontsize=28)
    # plt.title("Trade-off between stitch and projection quality", fontsize=36)

    plt.savefig("rank-inverted.png", dpi=200, bbox_inches='tight')
    # plt.show()


if __name__ == "__main__":
    main()
