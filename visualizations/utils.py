import logging
import os
import re

import matplotlib.pyplot as plt

import numpy as np


def extract_acc(line):
    avg_start = line.find("Avg: ")
    acc = float(line[avg_start+5:-1])
    return acc


def str_matrix_to_numpy(str_matrix: list):
    num_tasks = len(str_matrix)
    acc_matrix = np.zeros((num_tasks, num_tasks))
    for t in range(num_tasks):
        accs = re.findall("\d+\.\d+", str_matrix[t])[:-1]
        if not accs:
            raise RuntimeError("bitch")
        for i in range(len(accs)):
            acc_matrix[t, i] = accs[i]
    return acc_matrix


def last_acc(str_matrix: list):
    num_tasks = len(str_matrix)
    acc_matrix = np.zeros((num_tasks))
    for t in range(num_tasks):
        accs = re.findall("\d+\.\d+", str_matrix[t])[-1]
        if not accs:
            raise RuntimeError("bitch")
        acc_matrix[t] = float(accs)
    return acc_matrix


def calculate_representation_strength(log, num_tasks):
    avg_eigenvals = []
    for t in range(num_tasks):
        eigenvals = re.findall("\d+", log[t])[1:]
        eigenvals = [int(e) for e in eigenvals]
        avg_eigenvals.append(sum(eigenvals) / len(eigenvals))

    return avg_eigenvals


def calculate_ranks(log):
    ranks = []
    for t in range(100):
        rank = int(re.findall("\d+", log[t])[1])
        ranks.append(rank)
    return ranks


def extract_det(log):
    dets = []
    for epoch in range(200):
        det = float(re.findall("\d+\.\d+", log[epoch])[4])
        dets.append(det)
    return dets

def extract_losses(log):
    ce, ac, kd = [], [], []
    for epoch in range(200):
        floats = re.findall("\d+\.\d+", log[epoch])
        ce.append(float(floats[0]))
        kd.append(float(floats[1]))
        ac.append(- float(floats[3]))
    return ce, ac, kd

def extract_mahalanobis(log):
    distances = re.findall("\d+\.\d+", log)
    distances = [float(d) for d in distances]
    return distances