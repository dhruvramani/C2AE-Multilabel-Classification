import os
import arff
import argparse
import numpy as np

def get_features(path, features_dim):
    file_content = arff.load(open(path, "r"))
    data=np.array(file_content['data'], dtype="float32")
    return data[:, : features_dim]

def get_labels(path, features_dim, labels_dim):
    file_content = arff.load(open(path, "r"))
    data=np.array(file_content['data'], dtype="float32")
    return data[: , features_dim : ]

def set_dims(dataset_path):
    with open(os.path.join(dataset_path, "count.txt"), "r") as f:
        return [int (i) for i in f.read().split("\n") if i != ""]

if __name__ == '__main__':
    # Convert and save arff files to numpy-pickles for faster data I/O.
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="bibtex", help="Name of the dataset")
    args = parser.parse_args()
    dataset = args.dataset
    features_dim, labels_dim = set_dims("./{}/".format(dataset))
    train_features, train_labels = get_features("./{}/{}-train.arff".format(dataset, dataset), features_dim), get_labels("./{}/{}-train.arff".format(dataset, dataset), features_dim, labels_dim)
    train_features.dump("./{}/{}-train-features.pkl".format(dataset, dataset))
    train_labels.dump("./{}/{}-train-labels.pkl".format(dataset, dataset))
    test_features, test_labels = get_features("./{}/{}-test.arff".format(dataset, dataset), features_dim), get_labels("./{}/{}-test.arff".format(dataset, dataset), features_dim, labels_dim)
    test_features.dump("./{}/{}-test-features.pkl".format(dataset, dataset))
    test_labels.dump("./{}/{}-test-labels.pkl".format(dataset, dataset))
