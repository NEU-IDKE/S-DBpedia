# S-DBpedia: A benchmark dataset for spatial knowledge graph completion



- A benchmark for Spatial Knowledge Graph Completion (**SKGC**) extracted from **DBpedia**.
- It can be used to evaluate Spatial Knowledge Graph Embedding or Completion methods.
- The S-DBpedia baseline dataset contains two types of datasets. 
  - Data scale: S-DBpedia_small, S-DBpedia_medium, S-DBpedia_large, S-DBpedia.
  - Data Sparsity: S-DBpedia_GT5E, S-DBpedia_GT10E, S-DBpedia_GT20E, S-DBpedia_GT50E.

**Dataset Statstics and Analysis**

|     Dataset      | Relation | Entity  | Training Set | Validation Set | Test Set |
| :--------------: | :------: | :-----: | :----------: | :------------: | :------: |
| S-DBpedia_small  |   174    | 241,167 |   227,213    |     5,945      |  5,976   |
| S-DBpedia_medium |   182    | 412,896 |   454,426    |     21,329     |  21,149  |
| S-DBpedia_large  |   189    | 648,866 |   908,853    |     69,602     |  69,096  |
|    S-DBpedia     |   189    | 878,779 |  1,817,706   |    194,560     | 194,869  |
|  S-DBpedia_GT5E  |   131    | 44,127  |   128,258    |     15,296     |  15,268  |
| S-DBpedia_GT10E  |   166    | 90,487  |   266,529    |     32,316     |  32,301  |
| S-DBpedia_GT20E  |   173    | 183,659 |   541,295    |     66,295     |  66,283  |
| S-DBpedia_GT50E  |   182    | 461,140 |  1,219,286   |    147,897     | 148,128  |

- Data scale: S-DBpedia_small, S-DBpedia_medium, S-DBpedia_large, S-DBpedia.
- Data Sparsity: S-DBpedia_GT5E, S-DBpedia_GT10E, S-DBpedia_GT20E, S-DBpedia_GT50E.



**Attribute information**

We extracted all attributes of entities in the dataset from **DBpedia**. It contains **text**, **numerical**, and **image** information.

The full dataset with attribute information is available in [Zenodo](https://doi.org/10.5281/zenodo.7431612).

The data in zenodo includes the dataset file (dataset name) and all attribute files of the entity (Attribute.tar.gz)


**Assessment of the benchmark quality**

Detailed instructions for data quality assessment can be viewed in [Baseline_Quality_Assessment_Instructions.md](./Baseline_Quality_Assessment_Instructions.md).


## Repository structure description
- `DataPrecessCode/`: This folder contains the detailed process of creating S-DBpedia.
- `Code/`: This folder contains the experimental code for knowledge graph completion in the paper and detailed real use cases.
- `Additional-Information/`: This folder contains detailed information about the data processing part of the paper.
- `Quality_Assessment_code/`: This folder contains code used to evaluate benchmark quality.

> dataset available: https://doi.org/10.5281/zenodo.7431612


## How to use these data for spatial knowledge graph completion tasks?

We conduct link prediction experiments on all datasets using five models: **TransE**, **DistMult**, **ConvE**, **TransE-GDR** and **SSLP**. We have provided detailed experimental use cases. 

The experimental code and related instructions can be viewed at [README](./Code/README.md) under `Code/`. The configurations for each of the models are given in the `Code/` diectory.


## Others

Additional details about the dataset can be found in another [README](./Additional-Information/README.md).
