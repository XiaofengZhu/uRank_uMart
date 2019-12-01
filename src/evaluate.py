"""Evaluate the model"""

import argparse
import logging
import os

import numpy as np
import tensorflow as tf

from model.utils import Params
from model.utils import set_logger
from model.evaluation import evaluate
from model.reader import input_fn
from model.reader import load_dataset_from_tfrecords
from model.modeling import model_fn


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
# loss
parser.add_argument('--loss_fn', default='grank', help="model loss function") # urank, urrank, ranknet, listnet, listmle, lambdarank, mdprank
# data
parser.add_argument('--data_dir', default='../data/OHSUMED/4', help="Directory containing the dataset") # OHSUMED, MQ2007, MSLR-WEB10K, MSLR-WEB30K
parser.add_argument('--tfrecords_filename', default='OHSUMED.tfrecords', help="Directory containing the dataset") # OHSUMED, MQ2007, MSLR-WEB10K, MSLR-WEB30K
parser.add_argument('--restore_from', default='best_weights',
                    help="Subdirectory of model dir or file containing the weights")

if __name__ == '__main__':
    # Set the random seed for the whole graph
    tf.set_random_seed(230)

    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)
    params.dict['loss_fn'] = args.loss_fn
    # Load the parameters from the dataset, that gives the size etc. into params
    json_path = os.path.join(args.data_dir, 'dataset_params.json')
    assert os.path.isfile(json_path), "No json file found at {}, run build.py".format(json_path)
    params.update(json_path)

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # # Get paths for tfrecords
    path_eval_tfrecords = os.path.join(args.data_dir, 'test_' + args.tfrecords_filename)

    # Create the input data pipeline
    logging.info("Creating the dataset...")
    eval_dataset = load_dataset_from_tfrecords(path_eval_tfrecords)

    # Create iterator over the test set
    eval_inputs = input_fn('test', eval_dataset, params)
    logging.info("- done.")

    # Define the model
    logging.info("Creating the model...")
    eval_model_spec = model_fn('eval', eval_inputs, params, reuse=False)
    logging.info("- done.")

    logging.info("Starting evaluation")
    evaluate(eval_model_spec, args.model_dir, params, args.restore_from)