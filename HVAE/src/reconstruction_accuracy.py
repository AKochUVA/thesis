from argparse import ArgumentParser

import numpy as np
import torch
from sklearn.model_selection import KFold
import editdistance

from hvae_utils import read_expressions_json, load_config_file, create_batch
from symbol_library import generate_symbol_library
from model import HVAE
from train import train_hvae


def one_fold(model, train, test, epochs, batch_size, verbose):
    train_hvae(model, train, epochs, batch_size, verbose)

    total_distance = []
    for i in range((len(test) // batch_size) + 1):
        batch = create_batch(test[(i*batch_size):((i+1)*batch_size)])
        latent = model.encode(batch)[0]
        pts = model.decode(latent)
        for j in range(len(pts)):
            total_distance.append(editdistance.eval(test[i*batch_size+j].to_list(notation="postfix"), pts[j].to_list(notation="postfix")))
    print(len(total_distance))
    return total_distance


def one_experiment(name, trees, input_dim, latent_dim, epochs, batch_size, verbose, seed,
                   smaller_dataset=False, examples=2000, n_splits=5, results_path=None):
    kf = KFold(n_splits=n_splits)
    distances = []
    for i, (train_idx, test_idx) in enumerate(kf.split(trees)):
        print(f"Fold {i + 1}")
        if smaller_dataset:
            np.random.seed(seed + i)
            torch.manual_seed(seed + i)
            inds = np.random.permutation(train_idx)
            inds = inds[:examples]
            train = [trees[i] for i in inds]
        else:
            train = [trees[i] for i in train_idx]

        test = [trees[i] for i in test_idx]
        model = HVAE(input_dim, latent_dim)
        distances.append(one_fold(model, train, test, epochs, batch_size, verbose))
        print(f"Mean: {np.mean(distances[-1])}, Var: {np.var(distances[-1])}")
        print()
    fm = [np.mean(d) for d in distances]
    if results_path is not None:
        with open(results_path, "a") as file:
            file.write(f"{name}\t Mean: {np.mean(fm)}, Std dev: {np.std(fm)}, All: {', '.join([str(f) for f in fm])}\n")
    print(f"Mean: {np.mean(fm)}, Std dev: {np.std(fm)}, All: {', '.join([str(f) for f in fm])}")
    return fm


if __name__ == '__main__':
    parser = ArgumentParser(prog='Expression reconstruction', description='Evaluate the reconstruction ability of HVAE')
    parser.add_argument("-config", default="../configs/test_config.json")
    args = parser.parse_args()

    config = load_config_file(args.config)
    expr_config = config["expression_definition"]
    es_config = config["expression_set_generation"]
    training_config = config["training"]
    reconstruction_config = config["reconstruction"]

    if training_config["seed"] is not None:
        np.random.seed(training_config["seed"])
        torch.manual_seed(training_config["seed"])

    sy_lib = generate_symbol_library(expr_config["num_variables"], expr_config["symbols"], expr_config["has_constants"])
    HVAE.add_symbols(sy_lib)

    trees = read_expressions_json(es_config["expression_set_path"])

    one_experiment(es_config["expression_set_path"], trees, len(sy_lib), training_config["latent_size"],
                   training_config["epochs"], training_config["batch_size"], training_config["verbose"],
                   training_config["seed"], reconstruction_config["smaller_dataset"], reconstruction_config["num_examples"],
                   reconstruction_config["n_folds"], reconstruction_config["results_path"])
