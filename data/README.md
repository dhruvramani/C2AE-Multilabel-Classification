# Data Preparation
For faster I/O we load the `.arff` files and convert them to numpy arrays and dump them.

In order to run the script, you need to manually prepare the dataset in the following way :

After downloading the dataset, create a file named `count.txt` in the directory of the dataset and fill it in the following format :

> ```
> number_of_features
> number_of_labels
> ```

Run `to_numpy.py` in the following manner :
```shell
python3 to_numpy.py --dataset dataset_name
```
After the data is processed, you are all set to run the model!