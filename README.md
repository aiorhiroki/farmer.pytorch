# Pytorch segmentation

```bash
docker build -t pyroch_seg:latest .

docker run \
    --gpus all \
    -itd \
    --ipc=host \
    --shm-size=24g \
    -v /mnt:/mnt \
    --name cowboy \
    pyroch_seg:latest

docker exec -it cowboy bash -c \
    "cd $PWD && env CUDA_VISIBLE_DEVICES=0 python train.py"
```
