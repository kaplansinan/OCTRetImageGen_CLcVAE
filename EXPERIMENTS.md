### EXPERIMENTS

In this part, we provide details regarding how to conduct experiments including Exploratory data analysis(EDA), contrastive learning training and conditional variational autoencoder training. Follow the below instructions step by step to run experiments succesfully. 


#### Step 1 - Run Docker container 

* After installating the docker container,  navigate to **Docker** directory by `cd Docker`. 
* In `run.sh`, edit **DATA_DIR** path (data input path) and **MODEL_REG_DIR** path (where the models and outputs will be saved)
* Once the paths are ready, run the file by `sh run.sh` or `./run.sh`. This will start jupyter notebook server. You can access it by running `localhost:61499` on your browser. 
* On your browser, navigate to **notebooks** directory, where you will find notebooks to conduct experiments. 

#### Step 2 - Run Exploratory data analysis(EDA)
We provide two jupyter notebooks for EDA, which are: 

* **raw_train_data_eda.ipynb**: domanstrates the exploratory data analysis conducted for  all training set. 
* **filtered_train_data_eda.ipynb**: domanstrates the exploratory data analysis conducted for filtered training set by the representative sampling algortihm given in the supplementary material. 

Follow instrcutions in the notebooks for EDA. All the code is self-explanatory and commented as needed. 

#### Step 3 - Train contrastive learning model
Follow instructions in the below notebooks for training contrastive learning model. All the code is self-explanatory and commented as needed.

* **contrastive_model_training.ipynb**: domanstrates the training of contrastive learning model and extracting embeddings from filtered training set for the next step. 

#### Step 4 - Train conditional variational autoencoder(cVAE) model
Follow instructions in the below notebooks for training cVAE model. All the code is self-explanatory and commented as needed.

* **conditional_variational_autoencoder_training.ipynb**: presents the training cVAE model step by step and generating new samples after each training run. As an example, we present the training and data generation only for CNV class but one may easily train and generate samples for other classes by following the instructions in the notebook. 

