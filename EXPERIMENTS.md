In this part, we provide details regarding how to conduct experiments including data eda, contrastive learning training and conditional variationalautoencoder training.  After installating the docker container, edit the data paths and where the image will be saved in `run_docker.sh`.

#### Run Dococker container`run_docker.sh`

 After installating the docker container, edit the data paths and where the image will be saved in `run_docker.sh`.

 Afterwards. run the docker container by run docker.sh' '

 This will start jupyter notebook on port 64399. Then go to localhost:64399 in your browser to access jupyter notebooks.

 ### Run experiments  i
 All the notebooks related to explortory data analysis and training can be found under notebooks dir. Simply navigate the notebooks and start each notebook.

 1- run exploratory data analysis: 
 - drtrt.ipynb domanstrates the exploratory data analysis conducted for training set
 - drtrt.ipynb domanstrates the exploratory data analysis conducted for filtered training set

 2- train contrastive learning model 
 - drtrt.ipynb domanstrates the training and extracting embeddings from training set for the next step. One can 


3- train variational autoencoder model 
 - drtrt.ipynb: presents the training procecude step by step and generating new samples after each training run. 
