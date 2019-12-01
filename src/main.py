# rm *.txt & ./bash.sh
# /Users/xiaofengzhu/Documents/GitHub/uRank_urRank/uRank_urRank/src
import argparse
import logging
import os
import time
import tensorflow as tf
from model.utils import Params
from model.utils import set_logger
from model.training import train_and_evaluate
from model.reader import load_dataset_from_tfrecords
from model.reader import input_fn
from model.modeling import model_fn


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
# loss
parser.add_argument('--loss_fn', default='grank', help="model loss function") # rrank, urrank, ranknet, listnet, listmle, lambdarank, mdprank
# data
parser.add_argument('--data_dir', default='../data/OHSUMED/4', help="Directory containing the dataset") # OHSUMED, MQ2007, MSLR-WEB10K, MSLR-WEB30K
parser.add_argument('--tfrecords_filename', default='OHSUMED.tfrecords', help="Directory containing the dataset") # OHSUMED, MQ2007, MSLR-WEB10K, MSLR-WEB30K
parser.add_argument('--restore_dir', default=None, # experimen3s/base_model/best_weights
                    help="Optional, directory containing weights to reload before training")
# python main.py --restore_dir experiments/base_model/best_weights


if __name__ == '__main__':
    tf.reset_default_graph()
    # Set the random seed for the whole graph for reproductible experiments
    tf.set_random_seed(230)

    # Load the parameters from the experiment params.json file in model_dir
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)
    params.dict['loss_fn'] = args.loss_fn

    # # Load the parameters from the dataset, that gives the size etc. into params
    json_path = os.path.join(args.data_dir, 'dataset_params.json')
    assert os.path.isfile(json_path), "No json file found at {}, run prepare_data.py".format(json_path)
    params.update(json_path)

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    path_train_tfrecords = os.path.join(args.data_dir, 'train_' + args.tfrecords_filename)
    path_eval_tfrecords = os.path.join(args.data_dir, 'eval_' + args.tfrecords_filename)

    # Create the input data pipeline
    logging.info("Creating the datasets...")
    train_dataset = load_dataset_from_tfrecords(path_train_tfrecords)
    eval_dataset = load_dataset_from_tfrecords(path_eval_tfrecords)

    # Specify other parameters for the dataset and the model

    # Create the two iterators over the two datasets
    train_inputs = input_fn('train', train_dataset, params)
    eval_inputs = input_fn('vali', eval_dataset, params)
    logging.info("- done.")

    # Define the models (2 different set of nodes that share weights for train and validation)
    logging.info("Creating the model...")
    train_model_spec = model_fn('train', train_inputs, params)
    eval_model_spec = model_fn('vali', eval_inputs, params, reuse=True)
    logging.info("- done.")

    # Train the model
    # log tim
    start_time = time.time()
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(train_model_spec, eval_model_spec, args.model_dir, params, args.restore_dir)
    print("--- %s seconds ---" % (time.time() - start_time))   