### Download and Extract Training/Test Set 
The data used for experiments in the paper can be downloaded via following instructions. 

1. Download data by running `download_data.sh`(it might take time to download the data due to its size (5GB)). 
2. Once the data is ready, you may proceed to [EXPERIMENTS.md](EXPERIMENTS.md). 

We provide two csv files under the **data** directory, which are:
1. train.csv : contains paths of each image and label information
2. filter_train.csv: contains path  and label information of filtered images by representative sampling method described in supplementary material. 
3. test.cvs: contains path and label information of the test set

The given csv files will be used in experiments to train models. 
