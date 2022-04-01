# Impact of Evaluation Methodologies on Code Summarization

This repo hosts the code and data for the following ACL 2022 paper:

Title: [Impact of Evaluation Methodologies on Code Summarization][paper-utcs]

Authors: [Pengyu Nie](http://cozy.ece.utexas.edu/~pynie/), [Jiyang Zhang](https://jiyangzhang.github.io/), [Junyi Jessy Li](https://jessyli.com/), [Raymond J. Mooney](https://www.cs.utexas.edu/~mooney/), [Milos Gligoric](http://users.ece.utexas.edu/~gligoric/)

```bibtex
@inproceedings{NieETAL22EvalMethodologies,
  title =        {Impact of Evaluation Methodologies on Code Summarization},
  author =       {Pengyu Nie and Jiyang Zhang and Junyi Jessy Li and Raymond J. Mooney and Milos Gligoric},
  pages =        {to appear},
  booktitle =    {Annual Meeting of the Association for Computational Linguistics},
  year =         {2022},
}
```

## Introduction

This repo contains the code and data for producing the experiments in
[Impact of Evaluation Methodologies on Code
Summarization][paper-utcs]. In this work, we study the impact of
evaluation methodologies, i.e., the way people split datasets into
training, validation, and test sets, in the field of code
summarization. We introduce the time-segmented evaluation methodology,
which is novel to the code summarization research community, and
compare it with the mixed-project and cross-project methodologies that
have been commonly used.

The code includes:
* a data collection tool for collecting (method, comment) pairs with
  timestamps.
* a data processing pipeline for splitting a dataset following the
  three evaluation methodologies.
* scripts for running four recent machine learning models for code
  summarization and comparing their results across methodologies.

**How to...**
* **reproduce the training and evaluation of ML models on our
  collected dataset**: [install dependency][sec-dependency], [download
  all data][sec-downloads], and follow the instructions
  [here][sec-traineval].
* **reproduce our full study from scratch**: [install
  dependency][sec-dependency], [download `_work/src`][sec-downloads]
  (the source code for the ML models used in our study), and follow
  the instructions to [collect data][sec-collect], [process
  data][sec-process], and [train and evaluate models][sec-traineval].


## Table of Contents

1. [Dependency][sec-dependency]
2. [Data Downloads][sec-downloads]
3. [Code for Collecting Data][sec-collect]
4. [Code for Processing Data][sec-process]
5. [Code for Training and Evaluating Models][sec-traineval]

## Dependency
[sec-dependency]: #dependency

Our code require the following hardware and software environments.

* Operating system: Linux (tested on Ubuntu 20.04)
* Minimum disk space: 4 GB
* Python: 3.8
* Java: 8
* Maven: 3.6.3
* Anaconda/Miniconda: appropriate versions for Python 3.8 or higher

Additional requirements for training and evaluating ML models:

* GPU: NVIDIA GTX 1080 or better
* CUDA: 10.0 ~ 11.0
* Disk space: 2 GB per trained model

[Anaconda](https://www.anaconda.com/products/individual#Downloads) or
[Miniconda](https://docs.conda.io/en/latest/miniconda.html) is
required for installing the other Python library dependencies.  Once
Anaconda/Miniconda is installed, you can use the following command to
setup a virtual environment, named `tseval`, with the Python library
dependencies installed:

```
cd python/
./prepare_conda_env.sh
```

And then use `conda activate tseval` to activate the created virtual
environment.

The Java code `collector` will automatically be compiled as needed in
our Python code.  The Java library dependencies are automatically
downloaded, by the Maven build system, during this process.


## Data Downloads
[sec-downloads]: #data-downloads

All our data is hosted on UTBox via [a shared folder](https://utexas.box.com/s/32qq85ttp9js0qqnv68ebhvr19ajo7v9).

Data should be downloaded to this directory with the same directory
structure (e.g., `_work/src` from the shared folder should be
downloaded as `_work/src` under current directory).


## Code for Collecting Data
[sec-collect]: #code-for-collecting-data

### Collect the list of popular Java projects on GitHub

```
python -m tseval.main collect_repos
python -m tseval.main filter_repos
```

Results are generated to `results/repos/`:

* `github-java-repos.json` is the full list of projects returned by
  the GitHub API.
  
* `filtered-repos.json` is the list of projects filtered according to
  the conditions in our paper.
  
* `*-logs.json` documents the time, configurations, and metrics of the
  collection/filtering of the list.

Note that the list of projects may already differ from the list of
projects we used, because old projects may be removed, and the
ordering of projects may change.

### Collect raw dataset

Requires the list of projects at `results/repos/filtered-repos.json`

```
python -m tseval.main collect_raw_data
```

Results are generated to `_raw_data/`.  Each project's raw data
is in a directory named `$user_$repo` (e.g., `apache_commons-codec`):

* `method-data.json` is the list of method samples (includes code, API
  comments, etc.) extracted from the project at the selected revisions
  (at Jan 1st of 2018, 2019, 2020, 2021).
  
* `revision-ids.json` is the mapping from revision to the method
  samples that are available at that revision.
  
* `filtered-counters.json` is the count of samples discarded during
  collection according to our paper.
  
* `log.txt` is the log of the collection.

## Code for Processing Data
[sec-process]: #code-for-processing-data


### Process raw data to use our data structure (tseval.data.MethodData)

Requires the raw data at `_raw_data/`.

```
python -m tseval.main process_raw_data
```

Results are generated to `_work/shared/`:

* `*.jsonl` files are the dataset, where each file stores one field of
  all samples, and each line stores the field for one sample.
  
* `filtered-counters.json` is the combined count of samples discarded
  during collection.
  

### Apply methodologies (non task-specific part)

Requires the dataset at `_work/shared/`.

```
python -m tseval.main get_splits --seed=7 --split=Full
```

Results are generated to `_work/split/Full/`:

* `$X-$Y.json`, where X in {MP, CP, T} and Y in {train, val, test_standard}; and
  `$X1-$X2-test_common.json`, where X1, X2 in {MP, CP, T}.

  - each file contains a list of ids.
  
  - MP = mixed-project; CP = cross-project; T = temporally.
  
  - train = training; val = validation; 
    test_standard = standard test; test_common = common test.

### Apply methodologies (task-specific part)

Requires the dataset at `_work/shared/` and the splits at
`_work/split/Full/`.

From this point on, we define two variables to use in our commands:

* `$task`: to indicate the targeted code summarization task.
  - CG: comment generation.
  - MN: method naming.

* `$method`: to indicate the methodology used.
  - MP: mixed-project.
  - CP: cross-project.
  - T: temporally.

```
python -m tseval.main exp_prepare \
    --task=$task \
    --setup=StandardSetup \
    --setup_name=$method \
    --split_name=Full \
    --split_type=$method
# Example: python -m tseval.main exp_prepare \
#    --task=CG \
#    --setup=StandardSetup \
#    --setup_name=T \
#    --split_name=Full \
#    --split_type=T
```

Results are generated to `_work/$task/setup/$method/`:

* `data/` contains the dataset (jsonl files) and splits (ids in
  Train/Val/TestT/TestC sets).
  
* `setup_config.json` documents the configurations of this
  methodology.

## Code for Training and Evaluating Models
[sec-traineval]: #code-for-training-and-evaluating-models

### Prepare the Python environments for ML models

Requires Anaconda/Miniconda, and the models' source code at `_work/src/`.

```
python -m tseval.main prepare_envs --which=$model_cls
# Example: python -m tseval.main prepare_envs --which=TransformerACL20
```

Where the `$model_cls` for each model can be looked up in this table
(Transformer and Seq2Seq are using the same model class and
environment):

| $task | $model_cls         | Model         |
|:------|:-------------------|:--------------|
| CG    | DeepComHybridESE19 | DeepComHybrid |
| CG    | TransformerACL20   | Transformer   |
| CG    | TransformerACL20   | Seq2Seq       |
| MN    | Code2VecPOPL19     | Code2Vec      |
| MN    | Code2SeqICLR19     | Code2Seq      |

The name of the conda environment created is `tseval-$task-$model_cls`.

### Train ML models under a methodology

Requires the dataset at `_work/$task/setup/$method/`, and
activating the right conda environment
(`conda activate tseval-$task-$model_cls`).

```
python -m tseval.main exp_train \
    --task=$task \
    --setup_name=$method \
    --model_name=$model_cls \
    --exp_name=$exp_name \
    --seed=$seed \
    $model_args
# Example: python -m tseval.main exp_train \
#     --task=CG \
#     --setup_name=T \
#    --model_name=TransformerACL20 \
#    --exp_name=Transformer \
#    --seed=4182
```

Where `$exp_name` is the name of the output directory; `$seed` is the
random seed (integer) to control the random process in the experiments
(the `--seed=$seed` argument can be omitted for a random run using the
current timestamp as seed); `$model_args` is potential additional
arguments for the model and can be looked up in the following table:

| $task | Model         | $model_args    |
|:------|:--------------|:---------------|
| CG    | DeepComHybrid | (empty)        |
| CG    | Transformer   | (empty)        |
| CG    | Seq2Seq       | --use_rnn=True |
| MN    | Code2Vec      | (empty)        |
| MN    | Code2Seq      | (empty)        |

Results are generated to `_work/$task/exp/$method/$exp_name/`:

* `model/` the trained model.

* Other files documents the configurations for initializing and
  training the model.
  
### Evaluate ML models

Requires the dataset at `_work/$task/setup/$method/`, the trained
model at `_work/$task/exp/$method/$exp_name/`, and activating the
right conda environment (`conda activate tseval-$task-$model_cls`).

```
for $action in val test_standard test_common; do
    python -m tseval.main exp_eval \
        --task=$task \
        --setup_name=$method \
        --exp_name=$exp_name \
        --action=$action
done
# Example: for $action in val test_standard test_common; do
#    python -m tseval.main exp_eval \
#        --task=CG \
#        --setup_name=T \
#        --exp_name=Transformer \
#        --action=$action
#done
```

Results are generated to `_work/$task/result/$method/$exp_name/`:

* `$X_predictions.jsonl`: the predictions.

* `$X_golds.jsonl`: the golds (ground truths).

* `$X_eval_time.jsonl`: the time taken for the evaluation.

* Where $X in {val, test_standard, test_common-$method-$method1 (where $method1 != $method)}.


### Compute automatic metrics

Requires the evaluation results at
`_work/$task/result/$method/$exp_name/`, and the use of `tseval`
environment (`conda activate tseval`).

```
for $action in val test_standard test_common; do
    python -m tseval.main exp_compute_metrics \
        --task=$task \
        --setup_name=$method \
        --exp_name=$exp_name \
        --action=$action
done
# Example: for $action in val test_standard test_common; do
#    python -m tseval.main exp_compute_metrics \
#        --task=CG \
#        --setup_name=T \
#        --exp_name=Transformer \
#        --action=$action
#done
```

Results are generated to `_work/$task/metric/$method/$exp_name/`:

* `$X_metrics.json` and `$X_metrics.json`: the average of automatic metrics.

* `$X_metrics_list.pkl`: the (compressed) list of automatic metrics per sample.

* Where $X in {val, test_standard, test_common-$method-$method1 (where $method1 != $method)}.



[paper-arxiv]: https://arxiv.org/abs/2108.09619
[paper-utcs]: https://www.cs.utexas.edu/users/ai-lab/downloadPublication.php?filename=http://www.cs.utexas.edu/users/ml/papers/nie.acl2022.pdf&pubid=127948
