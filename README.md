# keras-image-captioning

## Preparation

### Download Dataset
```
./scripts/download_dataset.sh
```

### Download Pretrained Word Vectors
```
./scripts/download_pretrained_word_vectors.sh
```

### Download pycocoevalcap Data
```
./scripts/download_pycocoevalcap_data.sh
```

### Install the Dependencies
```
pip install -r requirements.txt
```

### Setup PYTHONPATH
```
source ./scripts/setup_pythonpath.sh
```

## Run a Training

For reproducing the final model, execute:
```
python -m keras_image_captioning.training \
  --training-label repro-final-model \
  --from-training-dir results/final-model
```

There are many arguments available that you can look inside `training.py`.

## Run an Inference and Evaluate It

```
python -m keras_image_captioning.inference \
  --dataset-type test \
  --method beam_search \
  --beam-size 3 \
  --training-dir var/flickr8k/training-results/repro-final-model
```

`dataset_type` can be either 'validation' or 'test'.

## License
MIT License. See LICENSE file for details.
