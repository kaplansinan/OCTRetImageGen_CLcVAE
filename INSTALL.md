## Development Stack

One may use Docker to conduct experiments described in the paper. To do so, simply follow the instructions below to install the development environment. 

### :whale: Docker Instructions


1. Change working directory to Docker: `cd Docker`

2. Build Docker image for development by running `build.sh`. 



Once the  image is built, the container called `tf2_trainer` is ready to be used for experiments.

**Note**: The installation is tested on Ubuntu 20.15 with  NVIDIA GeForce 3060 6GB GPU with CudaToolkit 11.5. To train models on GPU, make sure you have Nvidia drivers and Cuda Toolkit installed on your machine.
