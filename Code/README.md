

# How to use these data for spatial knowledge graph completion tasks?

We conduct link prediction experiments on all datasets using three models: **TransE**, **DistMult**, and **ConvE**. The following is the specific process of the experiment.

## Datasets

Available datasets are:

    S-DBpedia_small
    S-DBpedia_medium
    S-DBpedia_large
    S-DBpedia
    S-DBpedia_GT5E
    S-DBpedia_GT10E
    S-DBpedia_GT20E
    S-DBpedia_GT50E

Due to the large size of the data, we uploaded all the datasets to  [Zenodo](https://doi.org/10.5281/zenodo.7431612). You can download it from [Zenodo](https://doi.org/10.5281/zenodo.7431612) according to your needs.

Like other datasets, embedding models can use them directly.

## Reproduce experiment

### TransE

We use OpenKE's TransE code, and detailed information can be found at https://github.com/thunlp/OpenKE.

**Usage**

1. Install [PyTorch](https://pytorch.org/get-started/locally/)
2. Clone the OpenKE-PyTorch branch:

```shell
git clone -b OpenKE-PyTorch https://github.com/thunlp/OpenKE --depth 1
cd OpenKE
cd openke
```

3. Compile C++ files

```shell
bash make.sh
```

4. To run a model execute the following command :

```shell
# Save the Zenodo downloaded folder to OpenKE/benchmarks/
cd ../

# S-DBpedia
cp ../TransE/train_transe_S-DBpedia.py ./
python train_transe_S-DBpedia.py

# S-DBpedia_GT5E
cp ../TransE/train_transe_S-DBpedia_GT5E.py ./
python train_transe_S-DBpedia_GT5E.py

# S-DBpedia_GT10E
cp ../TransE/train_transe_S-DBpedia_GT10E.py ./
python train_transe_S-DBpedia_GT10E.py

# S-DBpedia_GT20E
cp ../TransE/train_transe_S-DBpedia_GT20E.py ./
python train_transe_S-DBpedia_GT20E.py

# S-DBpedia_GT50E
cp ../TransE/train_transe_S-DBpedia_GT50E.py ./
python train_transe_S-DBpedia_GT50E.py

# S-DBpedia_large
cp ../TransE/train_transe_S-DBpedia_large.py ./
python train_transe_S-DBpedia_large.py

# S-DBpedia_medium
cp ../TransE/train_transe_S-DBpedia_medium.py ./
python train_transe_S-DBpedia_medium.py

# S-DBpedia_small
cp ../TransE/train_transe_S-DBpedia_small.py ./
python train_transe_S-DBpedia_small.py

```



### DistMult

Like TransE, Distmult uses OpenKE code to conduct related experiments.

**Usage**

If TransE is already running normally, you can directly execute the following command, otherwise OpenKE needs to be configured. For the configuration of openKE, please see the TransE section above.

To run a model execute the following command :

```shell
# S-DBpedia
cp ../DistMult/train_distmult_S-DBpedia.py ./
python train_distmult_S-DBpedia.py

# S-DBpedia_GT5E
cp ../DistMult/train_distmult_S-DBpedia_GT5E.py ./
python train_distmult_S-DBpedia_GT5E.py

# S-DBpedia_GT10E
cp ../DistMult/train_distmult_S-DBpedia_GT10E.py ./
python train_distmult_S-DBpedia_GT10E.py

# S-DBpedia_GT20E
cp ../DistMult/train_distmult_S-DBpedia_GT20E.py ./
python train_distmult_S-DBpedia_GT20E.py

# S-DBpedia_GT50E
cp ../DistMult/train_distmult_S-DBpedia_GT50E.py ./
python train_distmult_S-DBpedia_GT50E.py

# S-DBpedia_large
cp ../DistMult/train_distmult_S-DBpedia_large.py ./
python train_distmult_S-DBpedia_large.py

# S-DBpedia_medium
cp ../DistMult/train_distmult_S-DBpedia_medium.py ./
python train_distmult_S-DBpedia_medium.py

# S-DBpedia_small
cp ../DistMult/train_distmult_S-DBpedia_small.py ./
python train_distmult_S-DBpedia_small.py
```



### ConvE 

**Usage**

1. Install [PyTorch](https://pytorch.org/get-started/locally/)
2. Install other requirements: `pip install -r requirements.txt`
3. Save the Zenodo downloaded folder to `ConvE/data/`
4. To run a model execute the following command :

```shell
# S-DBpedia_small
nohup python run.py -name Train-small -data S-DBpedia_small -batch 128 > ./log/train-S-DBpedia_small.log 2>&1 &

# S-DBpedia_medium
nohup python run.py -name Train-medium -data S-DBpedia_medium -batch 128 > ./log/train-S-DBpedia_medium.log 2>&1 &

# S-DBpedia_large
nohup python run.py -name Train-large -data S-DBpedia_large -batch 128 > ./log/train-S-DBpedia_large.log 2>&1 &

# S-DBpedia_GT5E
nohup python run.py -name Train-GT5E -data S-DBpedia_GT5E -batch 128 > ./log/train-S-DBpedia_GT5E.log 2>&1 &

# S-DBpedia_GT10E
nohup python run.py -name Train-GT10E -data S-DBpedia_GT10E -batch 128 > ./log/train-S-DBpedia_GT10E.log 2>&1 &

# S-DBpedia_GT20E
nohup python run.py -name Train-GT20E -data S-DBpedia_GT20E -batch 128 > ./log/train-S-DBpedia_GT20E.log 2>&1 &

# S-DBpedia_GT50E
nohup python run.py -name Train-GT50E -data S-DBpedia_GT50E -batch 128 > ./log/train-S-DBpedia_GT50E.log 2>&1 &
```

## Acknowledgement

We refer to the code of [OpenKE](https://github.com/thunlp/OpenKE). Thanks for their contributions.
