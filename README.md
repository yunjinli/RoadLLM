# RoadLLM

## Installation

```
git clone https://github.com/yunjinli/RoadLLM.git
cd RoadLLM

conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # Enable PEP 660 support.
pip install -e ".[train]"
```

## Preparing the Datasets

Please see [here](./docs/dataset.md).

## Acknowledgement

This project is heavily based on [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT).
