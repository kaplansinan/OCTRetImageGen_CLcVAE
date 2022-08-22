## Development Stack

One may use Docker to conduct experiments descibed in the paper. To do so, sinply follow the instruction below to install the development environment. 

### :whale: Docker Installation


1. Change working directory to Docker: `cd Docker`

2. Build Docker image for developement by running `build.sh`. 



Once the  image is built, the container called `tf2_trainer` is ready to be used for experiments.

**Note**: The installation is tested on Ubuntu 20.15 with  NVIDIA GeForce 3060 6GB GPU with CudaToolkit 11.5. To train models on GPU, make sure you have Nvidia drivers and CudaToolkit installed.