# Imports
import commentjson as cjson
import json
import time
import os.path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler, Dataset
from tqdm import tqdm

from HVAE.src.expression_set_generation import generate_grammar, generate_expressions
from HVAE.src.symbol_library import generate_symbol_library
from HVAE.src.tree import Node
from HVAE.src.hvae_utils import read_expressions_json
from HVAE.src.model import HVAE
from HVAE.src.train import train_hvae
from HVAE.src.symbolic_regression import read_eq_data, one_sr_run, check_on_test_set
from HVAE.src.evaluation import RustEval




def load_config_file(path):
    with open(path, "r") as file:
        jo = cjson.load(file)
    return jo


def generate_sr_expressions(symbols: list, num_variables: int, has_constants: bool, num_expressions: int,
                            max_tree_height: int, expression_set_path: str, filename: str = None, verbose: bool = True,
                            use_existing: bool = True):
    """Helper function to interface HVAE.src.expression_set_generation.py.
    Saves a new expression set in expression_set_path and returns the path. The filename is determined by input parameters, i.e.
    expr_3var+c_7depth.json represents an expression set with num_variables=3, has_constants=True and max_tree_height=7.
    """

    # Time logging
    start_time = time.time()

    # Create filename
    if filename is None:
        constant = '+c' if has_constants else ''
        filename = f"expr_{num_variables}var{constant}_{max_tree_height}depth.json"

    # Skip expression set generation if filename already exists
    if os.path.isfile(expression_set_path+filename) and use_existing:
        print(f"Existing expression set found at {expression_set_path+filename} will be used. \n "
              f"If you would like to create a new expression set, please set the use_existing parameter to False.")
        return expression_set_path+filename

    # Output
    if verbose:
        constants = 'with constants' if has_constants else 'without constants'
        print(f"Generating {num_expressions} expressions, with {num_variables} variables, {constants} and a maximum "
              f"tree height of {max_tree_height}. \n Expressions will be saved to {expression_set_path+filename} \n "
              f"Please note that this may take a while. You will be updated every {num_expressions/10} generated "
              f"expressions.")

    # Generate symbol library
    sy_lib = generate_symbol_library(num_vars=num_variables, symbol_list=symbols, has_constant=has_constants)
    Node.add_symbols(sy_lib)
    so = {s["symbol"]: s for s in sy_lib}

    # Optional (recommended): Generate training set from a custom grammar
    grammar = None

    if grammar is None:
        grammar = generate_grammar(sy_lib)

    # Generate expressions
    expressions = generate_expressions(grammar=grammar, number_of_expressions=num_expressions, symbol_objects=so,
                                       has_constants=has_constants, max_depth=max_tree_height)

    expr_dict = [tree.to_dict() for tree in expressions]

    # Save expressions
    save_path = expression_set_path + filename
    if save_path != "":
        with open(save_path, "w") as file:
            json.dump(expr_dict, file)

    # Output
    if verbose:
        print(f"Expression set generated. Took {round((time.time() - start_time) / 60, 2)} minutes.")

    return save_path


def train_HVAE_model(symbols: list, num_variables: int, has_constants: bool,
                     max_tree_height: int, expression_set_path, training_config, verbose: bool = True,
                     filename: str = None, use_existing: bool = True):
    """Helper function to interface HVAE.src.train.py.
    Trains a HVAE model and saves the parameters. The filename is determined by input parameters, i.e.
    params_3var+c_7depth.json represents an expression set with num_variables=1, has_constants=True and max_tree_height=7.
    """

    # Time logging
    start_time = time.time()

    # Read config files
    param_path = training_config["param_path"]

    # Create filename
    if filename is None:
        constant = '+c' if has_constants else ''
        filename = f"params_{num_variables}var{constant}_{max_tree_height}depth.json"

    # Skip model training if filename already exists
    if os.path.isfile(param_path + filename) and use_existing:
        print(f"Existing model parameters found at {param_path + filename} will be used. \n "
              f"If you would like to create a new model, please set the use_existing parameter to False.")
        return param_path + filename

    # Output
    if verbose:
        constants = 'with constants' if has_constants else 'without constants'
        print(f"Training HVAE model for expressions, with {num_variables} variables, {constants} and maximum tree height"
              f" {max_tree_height}. \n"
              f"Parameters will be saved to {param_path + filename} \n "
              f"Please note that this may take a while. Training is always verbose.")

    if training_config["seed"] is not None:
        np.random.seed(training_config["seed"])
        torch.manual_seed(training_config["seed"])

    sy_lib = generate_symbol_library(num_vars=num_variables, symbol_list=symbols, has_constant=has_constants)
    HVAE.add_symbols(sy_lib)

    trees = read_expressions_json(expression_set_path)

    model = HVAE(len(sy_lib), training_config["latent_size"])

    train_hvae(model, trees, training_config["epochs"], training_config["batch_size"], training_config["verbose"])

    # Save model
    save_path = param_path + filename
    torch.save(model, save_path)

    # Output
    if verbose:
        print(f"HVAE model training completed. Took {round((time.time() - start_time) / 60, 2)} minutes.")
        print(f"Saved trained HVAE model to {save_path}. ")

    return save_path


def run_symbolic_regression(config, expression_definition_config, expression_generation_config,
                            symbolic_regression_config, params_path,
                            train_set_path, test_set_path, dataset_name: str,
                            results_filename: str = None):
    """Helper function to interface HVAE.src.symbolic_regression
    Runs a symbolic regression according to the configuration.
    """

    train_set = read_eq_data(train_set_path)
    re_train = RustEval(train_set,
                        default_value=symbolic_regression_config["default_error"],
                        classification=symbolic_regression_config["classification"],
                        threshold=symbolic_regression_config["threshold"])

    sy_lib = generate_symbol_library(expression_definition_config["num_variables"],
                                     expression_definition_config["symbols"],
                                     expression_definition_config["has_constants"])
    so = {s["symbol"]: s for s in sy_lib}
    HVAE.add_symbols(sy_lib)
    model = torch.load(params_path)

    results = []
    for baseline in symbolic_regression_config["baselines"]:
        for i in range(symbolic_regression_config["number_of_runs"]):
            if symbolic_regression_config["seed"] is not None:
                seed = symbolic_regression_config["seed"] + i
            else:
                seed = np.random.randint(np.iinfo(np.int64).max)
            print()
            print("---------------------------------------------------------------------------")
            print(f"     Baseline: {baseline}, Run: {i + 1}/{symbolic_regression_config['number_of_runs']}")
            print("---------------------------------------------------------------------------")
            print()
            results.append(one_sr_run(model, config, baseline, re_train, seed))

    test_set = read_eq_data(test_set_path)
    re_test = RustEval(test_set,
                       default_value=symbolic_regression_config["default_error"],
                       classification=symbolic_regression_config["classification"],
                       threshold=symbolic_regression_config["threshold"])
    for i in range(len(results)):
        results[i] = check_on_test_set(results[i], re_test, so)

    # Create results_path
    if results_filename is None:
        constant = '+c' if expression_definition_config['has_constants'] else ''
        num_vars = expression_definition_config['num_variables']
        max_tree_height = expression_generation_config['max_tree_height']
        results_filename = f"{dataset_name}_{num_vars}var{constant}_{max_tree_height}depth.json"

    results_path = symbolic_regression_config["results_path"] + results_filename

    with open(results_path, "w") as file:
        json.dump(results, file)

    return results_path


def save_train_test_data(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series):
    """Function to save training and testing data together for Symbolic Regression."""
    # Create train and test csvs for EDHiE
    EDHiE_train = pd.concat([X_train, y_train], axis=1)
    EDHiE_test = pd.concat([X_test, y_test], axis=1)

    # Turn boolean columns into int columns
    boolean_cols = EDHiE_train.select_dtypes(include=['bool']).columns
    EDHiE_train[boolean_cols] = EDHiE_train[boolean_cols].astype(int)
    EDHiE_test[boolean_cols] = EDHiE_test[boolean_cols].astype(int)

    # Save to csv in correct format (no index or header)
    path = "./data/train_test_data/"
    num_vars = len(X_train.columns)
    filename = f"telco_{num_vars}vars"

    train_set_path = path + filename + "_train.csv"
    test_set_path = path + filename + "_test.csv"

    EDHiE_train.to_csv(path_or_buf=train_set_path, header=False, index=False)
    EDHiE_test.to_csv(path_or_buf=test_set_path, header=False, index=False)

    return train_set_path, test_set_path