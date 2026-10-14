## [MGKL: Mastery of a Three-Word Language](https://openreview.net/forum?id=eqMNwXvOqn)
![](https://img.shields.io/badge/version-1.0.0-blue)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/zjukg/MKGL)
[![Pytorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-orange)](https://huggingface.co/)
[![NeurIPS 2024](https://img.shields.io/badge/NeurIPS-2024-%23bd9f65?labelColor=%23bea066&color=%23ffffff)](https://neurips.cc/)

Welcome to the repository for the model of KG Language (MKGL). This project investigates the potential of LLMs in understanding and interacting with knowledge graphs, a domain that has received limited exploration in the context of NLP.


<div align="center">
    <img src="https://github.com/zjukg/MKGL/blob/master/imgs/arch.jpg" width="85%" height="auto" />
</div>

### Overview
Large language models (LLMs) have significantly advanced performance across a spectrum of natural language processing (NLP) tasks. Yet, their application to knowledge graphs (KGs), which describe facts in the form of triplets and allow minimal hallucinations, remains an underexplored frontier. In this project, we investigate the integration of LLMs with KGs by introducing a specialized KG Language (KGL), where a sentence precisely consists of an entity noun, a relation verb, and ends with another entity noun. 

<div align="center">
    <img src="https://github.com/zjukg/MKGL/blob/master/imgs/klora.jpg" width="85%" height="auto" />
</div>


### Environment

To run this project, please first install all required packages:

```
pip install --upgrade pandas transformers peft==0.9 bitsandbytes swifter deepspeed easydict pyyaml
```

please kindly install the pyg packages via wheels, which is much faster:

```
pip install --find-links MKGL/pyg_wheels/ torch-scatter torch-sparse torchdrug
```

The installed (not required) package details can be found in env.yaml.

### Preprocessing

Then, we need to preprocess the datasets,

for standard KG completion:

```
python preprocess.py -c config/fb15k237.yaml

python preprocess.py -c config/wn18rr.yaml
```

for inductive setting:

```
python preprocess.py -c config/fb15k237_ind.yaml --version v1

python preprocess.py -c config/wn18rr_ind.yaml --version v1
```

### Run with Single GPU

If you only has one GPU (better has 80GB memory under the default setting), please run the model with the following command:

```
python main.py -c config/fb15k237.yaml
```

### Run with Multiple GPU

If you can access multiple GPUs, please run the model with the following command:

```
accelerate launch --gpu_ids 'all' --num_processes 8 --mixed_precision bf16 main.py -c config/fb15k237.yaml
```

### Run with script

Please kindly use the provide scripts to run the model:

```
sh scripts/fb15k237.sh
```

### Baseline environment and Slurm checkpoint training

From the MKGL repository directory, install the environment described in
`instruction.txt` (without the Conda download / terms-of-service commands):

```bash
bash scripts/setup_mkgl.sh
sbatch scripts/train_baseline.sh
```

The job uses partition `gpu-long`, a five-hour limit, and one 80 GB A100
(`--gres=gpu:a100-80:1`, matching this cluster's GRES type). It defaults
to FB15k-237 transductive and preprocesses the dataset if its cache is missing.
Other runs can be submitted with:

```bash
sbatch scripts/train_baseline.sh config/fb15k237_ind.yaml v1
sbatch scripts/train_baseline.sh config/wn18rr.yaml
```

Load Conda before submitting and provide Hugging Face model access through your
existing login or exported `HF_TOKEN`. `MKGL_ENV` overrides the environment name;
`MKGL_DIR` overrides the repository location. Logs go to `logs/` and Slurm's
`slurm-mkgl-baseline-<jobid>.out/.err` files.
The script preserves `HF_HOME` (and your saved login token); only model downloads
are redirected through `HF_HUB_CACHE`, defaulting to `data/hf_cache/hub`.

If an existing environment reports `unsupported GNU version` or missing
`cusparse.h` while building TorchDrug, repair it without recreating the environment:

```bash
conda install -n mkgl -c nvidia/label/cuda-11.7.1 -c nvidia 'cuda-toolkit=11.7' -y
conda install -n mkgl -c conda-forge 'gcc_linux-64=11' 'gxx_linux-64=11' libxcrypt -y
```

The updated job script explicitly selects the Conda GCC/G++ compilers, includes
the CUDA headers, and builds/tests TorchDrug's CUDA kernel before loading Llama.
`MKGL_CC`, `MKGL_CXX`, and `MKGL_CUDA_HOME` can override these toolchain locations.
`libxcrypt` supplies `include/crypt.h`, which Python 3.8's headers may require
when compiling a C++ extension with the Conda compiler sysroot. For that missing
header alone, install `conda install -n mkgl -c conda-forge libxcrypt -y`.

The submit script evaluates on validation data. Compact LoRA, context-retriever,
and score-retriever weights are saved to `outputs/<config_name>/mkgl_checkpoint.pt`
every 500 optimizer steps, at epoch boundaries, and on normal completion.
`mkgl_best.pt` holds the best validation-MRR snapshot after training has started.
Set `CHECKPOINT_STEPS` to change the save interval.

The checkpoint includes configuration, KG token / vocabulary mapping, progress,
and a `training_completed` flag. The tokenizer is saved alongside it. Recreate
the same pretrained LLM and MKGL modules, load `payload["state_dict"]` into the
PEFT-wrapped MKGL model, then freeze its parameters for the RL stage. These
compact files are conditioner weights, not full optimizer-resume checkpoints.

### Cite

Please condiser citing our paper if it is helpful to your work!

```bigquery
@inproceedings{MKGL,
  author       = {Lingbing Guo and
                  Zhongpu Bo and 
                  Zhuo Chen and 
                  Yichi Zhang and
                  Jiaoyan Chen and
                  Lan Yarong and
                  Mengshu Sun and
                  Zhiqiang Zhang and
                  Yangyifei Luo and
                  Qian Li and
                  Qiang Zhang and
                  Wen Zhang and
                  Huajun Chen},
  title        = {MGKL: Mastery of a Three-Word Language},
  booktitle    = {{NeurIPS}},
  year         = {2024}
}
```

### Thanks

We appreciate [LLaMA](https://github.com/facebookresearch/llama), [Huggingface Transformers](https://github.com/huggingface/transformers/tree/main/src/transformers/models/llama), [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html), [Alpaca-LoRA](https://github.com/tloen/alpaca-lora), and many other related works for their open-source contributions.

