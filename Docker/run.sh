MAIN_DIR=$(pwd)/..
DATA_DIR=/home/hodor/dev/GITHUB/OCTRetImageGen_CLcVAE/data/
MODEL_REG_DIR=/home/hodor/dev/GITHUB/OCTRetImageGen_CLcVAE/reports/

docker run \
  --gpus all \
  -v $MAIN_DIR:/main \
  -v $DATA_DIR:/data/ \
  -v $MODEL_REG_DIR:/model_registry \
  -p 61499:61499 \
  -p 6006:6006 \
  --shm-size 8G \
  --rm \
  -it tf2_trainer:latest \
  bash start_jupyter_notebook.sh
