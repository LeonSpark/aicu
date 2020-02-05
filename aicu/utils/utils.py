# -*- encoding: utf-8 -*-
import random
import numpy as np
import torch


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(task_name, preds, labels):
    if task_name == 'sequence_classification':
        return {'acc':  simple_accuracy(preds, labels)}


def simple_accuracy(preds, labels):
    return (preds == labels).mean()
