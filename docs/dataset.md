# Prepare the Datasets

We use Waymo, BDD100k, nuImages, Cityscapes, Mapillary, R2S100k, RDD.

## Step1: Download the datasets

Please follow the steps below to download the datasets and formulate your dataset directory like:

```
<path/to/dataset>/
├── waymo
├── bdd100k
├── nuimages
├── cityscapes
├── mapillary
├── r2s100k
└── RDD2022

```

### Waymo

First, clone the [repo](https://github.com/yunjinli/waymo-open-dataset-RGB-semseg) and
follow the steps [here](https://github.com/yunjinli/waymo-open-dataset-RGB-semseg) to create a conda environment, and download the train/val split. Then, run the following commands, this will download the `*.parquet` from the gdrive and then extract only the RGB, instance, and semantic masks:

```
cd <path/to/waymo-open-dataset-RGB-semseg>
python process_image_seg.py --mode training --output_dir <path/to/dataset>/waymo/
python process_image_seg.py --mode validation --output_dir <path/to/dataset>/waymo/
```

### BDD100k

Download the dataset from https://dl.cv.ethz.ch/bdd100k/data/.

```
./dataset_utils/extract_bdd100k.bash <path/to/bdd100k_*.zip> <path/to/dataset>/bdd100k
```

The structure of the directory should look like this:

```
<path/to/dataset>/bdd100k
├── images
|   └── 10k
|       ├── test
|       ├── train
|       └── val
└── labels
    ├── ins_seg
    ├── lane
    ├── pan_seg
    └── sem_seg
```

### nuImages

Download the dataset from https://www.nuscenes.org/nuimages#download

- `nuimages-v1.0-all-metadata.tgz`
- `nuimages-v1.0-all-samples.tgz`

```
./dataset_utils/extract_nuimages.bash <path/to/nuimages-v1.0-all-samples.tgz> <path/to/nuimages-v1.0-all-metadata.tgz> <path/to/dataset>/nuimages
```

### Cityscapes

Download the dataset from https://www.cityscapes-dataset.com/downloads/

- `gtFine_trainvaltest.zip`
- `leftImg8bit_trainvaltest.zip`
- `leftImg8bit_trainextra.zip`

```
./dataset_utils/extract_cityscape.bash <path/to/cityscapes*.zip> <path/to/dataset>/cityscapes
```

The structure of the directory should look like this:

```
<path/to/dataset>/cityscapes
├── gtFine
|   ├── test
|   ├── train
|   └── val
└── leftImg8bit
    ├── test
    ├── train
    ├── train_extra
    └── val
```

### Mapillary

Download the dataset from https://www.mapillary.com/dataset/vistas

- `mapillary-vistas-dataset_public_v2.0.zip`

```
./dataset_utils/extract_mapillary.bash <path/to/mapillary.zip> <path/to/dataset>/mapillary
```

### R2S100k

Request access from matifbutt@outlook.com and download the r2s100k dataset.

- `test.zip`
- `train.zip`
- `val.zip`
- `test_labels.zip`
- `train_labels.zip`
- `val_labels.zip`

Save these .zip to `<path/to/dataset>/r2s100k`

```
./dataset_utils/extract_r2s100k.bash <path/to/dataset>/r2s100k <path/to/dataset>/r2s100k
```

The structure of the directory should look like this:

```
<path/to/dataset>/r2s100k
├── test
├── test_labels
├── train
├── Train-Labels
├── val
└── val_labels
```

### RDD

```
cd <path/to/dataset>
wget https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/RDD2022.zip
unzip RDD2022.zip -d ./RDD2022
unzip ./RDD2022/RDD2022_all_countries/Japan.zip -d ./RDD2022/RDD2022_all_countries
unzip ./RDD2022/RDD2022_all_countries/India.zip -d ./RDD2022/RDD2022_all_countries
unzip ./RDD2022/RDD2022_all_countries/Norway.zip -d ./RDD2022/RDD2022_all_countries
unzip ./RDD2022/RDD2022_all_countries/United_States.zip -d ./RDD2022/RDD2022_all_countries

cd <path/to/RoadLLM>
python dataset_utils/rdd_utils/output_ann_paths.py <path/to/dataset>/RDD2022/RDD2022_all_countries
python dataset_utils/rdd_utils/copy_train_val_images.py <path/to/dataset>/RDD2022/RDD2022_all_countries train <path/to/dataset>/RDD2022
python dataset_utils/rdd_utils/copy_train_val_images.py <path/to/dataset>/RDD2022/RDD2022_all_countries val <path/to/dataset>/RDD2022

touch <path/to/dataset>/RDD2022/RDD2022_all_countries/labels.txt
echo -e "D00\nD10\nD20\nD40" > <path/to/dataset>/RDD2022/RDD2022_all_countries/labels.txt

python dataset_utils/rdd_utils/voc2coco.py <path/to/dataset>/RDD2022/RDD2022_all_countries

mkdir -p <path/to/dataset>/RDD2022/annotations

python dataset_utils/rdd_utils/voc2coco.py --ann_paths_list <path/to/dataset>/RDD2022/RDD2022_all_countries/train_annotation_path.txt --annotation_ids <path/to/dataset>/RDD2022/RDD2022_all_countries/train_id.npy --labels <path/to/dataset>/RDD2022/RDD2022_all_countries/labels.txt --output <path/to/dataset>/RDD2022/annotations/instances_train2017.json

python dataset_utils/rdd_utils/voc2coco.py --ann_paths_list <path/to/dataset>/RDD2022/RDD2022_all_countries/val_annotation_path.txt --annotation_ids <path/to/dataset>/RDD2022/RDD2022_all_countries/val_id.npy --labels <path/to/dataset>/RDD2022/RDD2022_all_countries/labels.txt --output <path/to/dataset>/RDD2022/annotations/instances_val2017.json

```

## Step2: Generate dense captions

Configure [dataset_paths.json](../configs/dataset_paths.json) to apply to your local paths.

```
python dataset_utils/dense_captioning.py --dataset_name <r2s100k/bdd100k/rdd/cityscapes/cityscapes_extra/mapillary/nuimages/waymo> --output_root <path/to/dataset>/roadllm
```

## Step3: Generate pretrained.json

```
python dataset_utils/generate_dataset.py --image_base_path <path/to/dataset> --caption_base_path <path/to/dataset>/roadllm
```
