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

    upper_bounds = [i for i, x in enumerate(log) if "Mahalanobis per task:" in x]
    line = log[upper_bounds[-1]]
    mahalanobis_per_task = extract_mahalanobis(line)

    upper_bounds = [i for i, x in enumerate(log) if "Norm of pseudoprototypes" in x]
    line = log[upper_bounds[-1]]
    pseudo_norm = extract_mahalanobis(line)


    err1 = [345, 311, 287, 253, 215, 195, 167, 125, 91, 53]

    ax1.plot([_+1 for _ in range(10)], mahalanobis_per_task, 'o-', color="black", label="Rank", linewidth=3, markersize=10)
    ax1.errorbar([_+1 for _ in range(10)], mahalanobis_per_task, yerr=err1, fmt='o-', color="black")
    ax1.set_ylabel('Mahalanobis distance', color="black", fontsize=28)
    ax1.tick_params(axis='y', labelcolor="black")
    plt.yticks(fontsize=28)
    plt.xlabel("Task", fontsize=34)
    plt.xticks([_+1 for _ in range(10)], fontsize=28)

    err2 = [0.46, 0.42, 0.49, 0.51, 0.45, 0.54, 0.57, 0.58, 0.65, 0.65]
    ax2 = ax1.twinx()
    ax2.plot([_+1 for _ in range(10)], pseudo_norm, 'o-', color="tab:blue", label="Translation error (cm)", linewidth=3, markersize=10)
    ax2.errorbar([_+1 for _ in range(10)], pseudo_norm, yerr=err2, fmt='o-', color="tab:blue")
    ax2.set_ylabel('Norm of pseudoprototypes', color="tab:blue", fontsize=28)
    ax2.tick_params(axis='y', labelcolor="tab:blue")

    plt.yticks(fontsize=28)
    # plt.title("Trade-off between stitch and projection quality", fontsize=36)

    plt.savefig("distance-pseudo.png", dpi=200, bbox_inches='tight')
    # plt.show()


if __name__ == "__main__":
    main()
