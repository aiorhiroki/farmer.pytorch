# quick start

## download dataset

Segmentation
```
git clone https://github.com/alexgkendall/SegNet-Tutorial ./seg_data
```

## Train example
```
docker exec -it cowboy bash -c \
    "cd $PWD && env CUDA_VISIBLE_DEVICES=0 python train.py"
```