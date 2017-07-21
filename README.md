# C2AE

This is the Tensorflow implementation for the paper 'Learning Deep Latent Spaces for Multi-Label Classfications' published in AAAI 2017.

## Installation
The model was built and tested using Python 3!
Install the following dependencies : 
```shell
pip3 install liac-arff 
```
[Tensorflow](https://www.tensorflow.org/install/)

## Running
This code supports the `.arff` data format, however if you wish to use any other data format, convert it into numpy arrays and dump it to the `data/dataset_name` with the name format as mentioned in `data/README.md` and modify `model/src/parser.py`.

```shell
cd ./model/src
python3 __main__.py
```
# Logs
All the logs are saved in `./model/stdout` and you can visualize the loss using tensorboard by pointing it to `./model/results/tensorboard`.