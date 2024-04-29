# kgTransformer

This is the original implementation for KDD 2022 paper
_[Mask and Reason: Pre-Training Knowledge Graph Transformers for Complex Logical Queries](http://keg.cs.tsinghua.edu.cn/jietang/publications/KDD22-Liu-et-al-KG-Transformer.pdf)_.

## Prerequisites

* `pytorch>=1.7.1,<=1.9`
    * Note: Pytorch version greater than 1.9 has OOM bugs. See https://github.com/pytorch/pytorch/issues/67680.
* `pytorch-geometric`
    * See [PyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) implementation. For older PyTorch version, check out [here](https://data.pyg.org/whl/)
* `fastmoe`

Example installation using [`conda`](https://conda.io):

```bash
# Use the cuda version that matches your nvidia driver and pytorch
conda install "pytorch>=1.7.1,<=1.9" cudatoolkit=11.3 pyg -c pyg -c pytorch -y

# To compile fastmoe, CUDA `nvcc` toolchain is required.
# If not exists, it can be installed with conda:
conda install cudatoolkit-dev=11.3 "gxx_linux-64<=10" nccl -c conda-forge -y
# `nvcc` does not support gcc>10 as of 2022/06.

# Download fastmoe submodule if not already downloaded
git submodule update --init
cd fastmoe
pip install -e .
```

## Reproduction

The parameters in the paper is preloaded in [`configs/`](configs/).
Change `root_dir` option for the location to save model checkpoints.

Dataset can be downloaded from [http://snap.stanford.edu/betae/KG_data.zip](http://snap.stanford.edu/betae/KG_data.zip).
The location for the extracted dataset
should be specified in the `data_dir` in the config files.
For exmpale, if the `FB15k-237-q2b` dataset is in `/data/FB15k-237-q2b`,
this is what the `data_dir` options should be set.

Alternatively, pretrained models are available
at [OneDrive](https://1drv.ms/u/s!An61mxr_SiETgq9ZdWVZPMnYC35k2A?e=2s8FxH).

To reproduce all results for `FB15k-237`:

```bash
kgt="python main.py -c configs/fb15k-237.json"
# run pretrain1 & pretrain2
$kgt pretrain1 pretrain2
# Do multi-task finetuning for all tasks
$kgt reasoning_multi_1e5 kgt reasoning_multi_1e6
# Do single-task finetuning for each task
$kgt reasoning_1p reasoning_2p reasoning_3p
$kgt reasoning_2i reasoning_3i
$kgt reasoning_ip reasoning_pi
$kgt reasoning_2u reasoning_up
```

For `NELL995`:

```bash
kgt="python main.py -c configs/nell995.json"
# run pretrain
$kgt pretrain
# Do multi-task finetuning
$kgt reasoning_multi
# Do finetuning for tasks
$kgt reasoning_1p reasoning_2p reasoning_3p
$kgt reasoning_2i reasoning_3i
$kgt reasoning_ip reasoning_pi
$kgt reasoning_2u reasoning_up
```

## Documentation

### Model

There are two main implementations for kgTransformer,
both of which resides in [`model.py`](./model.py).

* `KGTransformer` is the original implementation that uses sparse operations to perform attention calculations, causing
  randomness beyond seed's control in the CUDA multi-thread parallel execution.
* `D_KGTransformer` is the alternative version that uses matrix attention to avoid reproducibility issues.

Our current implementation is based on `D_KGTransformer`.

### Training

Training-related utilities can be found in [`train.py`](./train.py).
They accept `Iterator`'s that yield batched data,
identical to the output of a `torch.utils.data.DataLoader`.
The most useful functions are `main_mp()` and `ft_test()`.

`TrainClient` scatters data onto different workers
and perform multi-GPU training based on `torch.nn.parallel.DistributedDataParallel`.
Example usage can be found in `main_mp()`.

### Config Files

Each config file is a JSON key-value mapping that maps a task name to a task.
The tasks can be run directly from the command line:

```bash
python main.py <task_name> [<task_name>...]
```

In a specific task, `base` option specifies the task it should inherit from.
`type` option specifies the type of operation of this configuration.
See [`main.py`](./main.py) for a full list of available options.

## Troubleshooting

<details>
<summary>NCCL Unhandled System Error</summary>

We observed that Infiniband is not supported by `fastmoe` on some machines.

NCCL with Infiniband can be disabled using an environment variable.

```bash
export NCCL_IB_DISABLE=1
```

</details>

<details>
<summary>CUDA Out of Memory</summary>

Adjust batch size and retry.
If the issue persists, downgrade pytorch to as early as possible (e.g. LTS 1.8.2 as of 2022/07).
This is possibly due to memory issues in higher pytorch versions.
See https://github.com/pytorch/pytorch/issues/67680 for more information.

</details>

## Citation

If you are interested in our work and wish to give us a credit,
you can use the following BibTeX:

```
@inproceedings{liu2022kgxfmr,
  title={Mask and Reason: Pre-Training Knowledge Graph Transformers for Complex Logical Queries},
  author={Liu, Xiao and Zhao, Shiyu and Su, Kai and Cen, Yukuo and Qiu, Jiezhong and Zhang, Mengdi and Wu, Wei and Dong, Yuxiao and Tang, Jie},
  booktitle={Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2022}
}
```
