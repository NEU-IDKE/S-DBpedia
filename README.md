# S-DBpedia: A benchmark dataset for spatial knowledge graph completion

- A benchmark for Spatial Knowledge Graph Completion (**SKGC**) extracted from **DBpedia**.
- It can be used to evaluate Knowledge Graph Embedding methods or to evaluate Knowledge Graph Embedding methods with **attributes**.
- The S-DBpedia baseline dataset contains two types of datasets. 
  - Data scale: S-DBpedia_small, S-DBpedia_medium, S-DBpedia_large, S-DBpedia.
  - Data Sparsity: S-DBpedia_GT5E, S-DBpedia_GT10E, S-DBpedia_GT20E, S-DBpedia_GT50E.

## Dataset Statstics and Analysis

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



## Attributes

We extracted all attributes of entities in the dataset from **DBpedia**. It contains **text**, **numerical**, and **image** information.

The full dataset with attribute information is available in [Zenodo](https://doi.org/10.5281/zenodo.7431612).



## Benchmarking

- Link predection experiments are conducted with three models **TransE**, **DistMult**, and **ConvE** on all datasets.
- the configurations for each of the models are given in the Code diectory.

## Others

Additional details about the dataset can be found in another [README](./Additional-Information/README.md).
