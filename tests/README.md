# quick start

## Prerequisite

- [install poetry](https://python-poetry.org/docs/#installation)
- install requirements
```
poetry install
```

### download dataset
```
git clone https://github.com/alexgkendall/SegNet-Tutorial ./seg_data
```

### Train example
```
nohup poetry run dvc repro &
tail train.out -f
```

### Visualize
[DVC Studio](https://studio.iterative.ai/user/aiorhiroki/views/farmer_pytorch-vf2yvhty6d)