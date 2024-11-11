# STM: A Spatio-Temporal Model for Dynamic Graph Fraud Detection

This repository contains the code and datesets used for the experiments described in the paper entitled with

_STM: A Spatio-Temporal Model for Dynamic Graph Fraud Detection_

![](.png)

## Requirements

The experiments were conducted on a 2 GHz Linux server with an RTX 4090D (24GB) 

## Datasets

Datasets: 

- [wikipedia](http://snap.stanford.edu/jodie/wikipedia.csv)

- [mooc](http://snap.stanford.edu/jodie/mooc.csv)

- [reddit](http://snap.stanford.edu/jodie/reddit.csv)

- [Dgraph-Fin](https://dgraph.xinye.com/dataset)


### Prerequisites

```
scikit-learn==1.5.2
pandas==1.1.0
torch==2.4.0+cu124
rapids-dask-dependency==24.10.00
```

Make sure that the environment is set up before the experiment.

### Model Training

The relevant datasets have been uploaded to the repository, and the code can be run directly.

```
python run_FTM_anomalousDetection.py -k 1 --data wikipedia --n_epoch 15 --embedding_module HopTransformer --use_time_line --time_line_length 2 --sample_mode time --hard_sample --n_layer 1 --bs 100 --use_ST_dist 
```

### Acknowledgment

This repo is built upon the following work:

```
TGN: Temporal Graph Networks  
https://github.com/twitter-research/tgn

FTM：A Frame-level Timeline Modeling Method for Temporal Graph Representation Learning
https://github.com/yeeeqichen/FTM
```

Many thanks to the authors and developers!

### Main Parameters Settings

|  **Parameter**   | **Type** |             **Description**             |
| :--------------: | :------: | :-------------------------------------: |
|       data       |   str    |              Dataset name               |
|        bs        |   int    |               Batch_size                |
|     n_degree     |   int    |      Number of neighbors to sample      |
|      n_head      |   int    | Number of heads used in attention layer |
| time_line_length |   int    |             Number of frame             |
|      epoch       |   int    | The number of rounds of model training  |
| embedding_module |   str    |        Type of embedding module         |

**The repository will be continuously updated**.
