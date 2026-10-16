# Track 2a - UZH - SwissGov-RSD: Detecting Cross-Lingual Semantic Differences in Swiss Government Websites

## Context & Background

according to the [official task description](https://hackapertus.notion.site/track-2a-uzh)

Swiss government websites are published in the official Swiss languages and English, and every language version is expected to carry equivalent information. In practice, translation errors and unsynchronised content updates introduce semantic differences between versions. SwissGov-RSD is a benchmark of 224 multi-parallel Swiss government documents in English, German, French and Italian, with human annotations of semantic differences for the EN–DE, EN–FR and EN–IT document pairs.

## Problem Description

according to the [official task description](https://hackapertus.notion.site/track-2a-uzh)

Given two related documents in different languages, predict for each token how strongly its meaning differs from the corresponding content in the other document. Systems should identify both symmetric differences and asymmetric ones such as additions and omissions.

## Structure

There are different pipelines that were used at different stages throughout the project:

```
src
├── evaluation
├── gpu_env (-> shared Dockerfile and requirements.txt for synthetic_data and train)
├── synthetic_data
├── train
└── train_4gpus_clariden
```

- `evaluation` is the one that brings it all together and returns a `comparison.csv` file to compare different model performances on recognizing semantic differences on the [SwissGov-RSD dataset](https://huggingface.co/datasets/ZurichNLP/SwissGov-RSD). It is executed by running `make run`.
- `synthetic_data` was used to build synthetic data for our fine-tuning runs. It is executed by running `make synthetic`.
- `train` uses the training data that was generated in `synthetic_data` (so `synthetic_data` has to be run first!) and creates a quantized LoRA adapter to improve Apertus' RSD abilities.
- `train_4gpus_clariden` uses the same training data from `synthetic_data` and creates a fully fine-tuned model. The set-up is specifically tailored to run on one Clariden node with 4 GH200 GPUs.

Because the pipelines for building the synthetic data and for training the adapter take several hours and our quantised LoRA adapter as well as the full fine-tuned model are accessible via HuggingFace, `make run` runs the evaluation pipeline only. The other pipelines can be run and assessed via the commands `make synthetic` and `make train`. There is no `make` command for `train_4gpus_clariden` because it can only be run on a Clariden node directly but we provide instructions to do so below.


## How To Run the `evaluation` Notebook

### Prerequisites

- CSCS Clariden compute allocation
- [ssh for CSCS](https://docs.cscs.ch/access/ssh/#adding-ela-as-a-jump-host-in-ssh-configuration) set up (`ssh clariden` works)
- Docker installed on your local machine and running

### Serving the model

#### Why we serve the model ourselves

- Avoid gateway timeouts: even with streaming the response, we could not avoid gateway timeouts (60s on https://api.inference.cscs.ch/v1 and https://api.swissai.svc.cscs.ch/v1) when running the evaluation.
- Therefore, we serve the model ourselves on the CSCS Clariden cluster via [model-launch](https://github.com/swiss-ai/model-launch):
  1. ssh into the Clariden terminal: `ssh clariden`
  2. Make sure that `model-launch` is available and set up in your `/users/<your-cscs-username>` directory according to their [README](https://github.com/swiss-ai/model-launch/blob/main/README.md).
  3. Switch into the directory (`cd model-launch`) and activate the environment: `source .venv/bin/activate`
  4. Execute the `sml advanced` command below in the terminal.
  5. A terminal UI (TUI) will open. You can leave it as is and continue with the next chapter "Setting up an ssh tunnel to the served model".


#### Apertus 1.5 8B with quantized LoRA-Adapter

This command serves the base Apertus 1.5 8B model (under `<your-cscs-username>/swiss-ai/Apertus-v1.5-8B`) as well as the quantized LoRA fine-tuned version with our adapter on top (under `swiss-ai/apertus1.5-8b-rsd-mixed-weighted`):

```bash
sml advanced \
  --tui \
  --partition debug \
  --nodes-per-replica 1 \
  --disable-opentela \
  --framework vllm \
  --time 00:45:00 \
  --environment src/swiss_ai_model_launch/assets/envs/vllm_apertus_1.5_release.toml \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-v1.5-8B \
    --served-model-name swiss-ai/Apertus-v1.5-8B \
    --enable-lora \
    --max-lora-rank 32 \
    --lora-modules swiss-ai/apertus1.5-8b-rsd-mixed-weighted=giandrin/apertus1.5-8b-rsd-mixed-weighted \
    --host 127.0.0.1 \
    --chat-template-content-format string \
    --data-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 60000 \
    --enable-auto-tool-choice \
    --tool-call-parser apertus"
```

#### Fully fine-tuned Apertus 1.5 8B

This command serves the fully fine-tuned version (under `<your-cscs-username>/swiss-ai/apertus1.5-8b-rsd-mixed-weighted-full-fine-tune`).

```bash
sml advanced \
  --tui \
  --partition debug \
  --nodes-per-replica 1 \
  --disable-opentela \
  --framework vllm \
  --time 00:45:00 \
  --environment src/swiss_ai_model_launch/assets/envs/vllm_apertus_1.5_release.toml \
  --framework-args "--model giandrin/apertus1.5-8b-rsd-mixed-weighted-full-fine-tune \
    --served-model-name swiss-ai/apertus1.5-8b-rsd-mixed-weighted-full-fine-tune \
    --dtype bfloat16 \
    --host 127.0.0.1 \
    --chat-template-content-format string \
    --data-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 60000 \
    --enable-auto-tool-choice \
    --tool-call-parser apertus"
```

### Setting up an ssh tunnel to the served model

After launching the model, open an ssh tunnel from your local machine to the node where the model is running. Change `<your-cscs-username>` to your CSCS username and `<node-id>` to the node id (of the form `nid00XXXX`) you see in the TUI after your job started running (may take a minute). Keep this terminal open and run `make run` in another terminal.

```bash
ssh -N -J <your-cscs-username>@ela.cscs.ch,<your-cscs-username>@clariden.alps.cscs.ch -L 8080:localhost:8080 <your-cscs-username>@<node-id>
```

### Start the run



If not already done, change to the `track_2a` directory.

```bash
cd track_2a/
```

All settings are managed through the `.env` file which is created the first time you run `make run` with default values (if you followed the steps above, it will prompt the fine-tuned model for the RSD values on the test split). If you wish to change any variables before the first `make run`, just `cp .env.example .env` and edit it.

``` bash
make run
```

If the model is still starting during your run, the notebook will automatically retry contacting your node every 10 seconds for 15 minutes until the node is up and running. This is normal and expected as the model needs some time to start (if the threshold of 15 minutes is reached without being able to connect, check if the node didn't error and that you set the ssh tunnel and try again):

```text
Executing Cell 16--------------------------------------
not ready (APIConnectionError), retrying in 10s
not ready (APIConnectionError), retrying in 10s
not ready (APIConnectionError), retrying in 10s
ready: ['gmacagni/swiss-ai/Apertus-v1.5-8B', 'swiss-ai/apertus1.5-8b-rsd-mixed-weighted']
```

### Results

All results are saved to `track_2a/src/evaluation/RSD_Validation_Pipeline_results`. Check out `track_2a/src/evaluation/RSD_Validation_Pipeline_results/comparison.csv` for a comparison between different runs.

## How to run the `synthetic_data` notebooks

### Prerequisites

- Linux with an NVIDIA GPU (tested on an RTX 5090 with 32 GB VRAM)
- NVIDIA driver 580 or newer
- Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### What it does

`make synthetic` builds the container and runs `src/synthetic_data/run.sh`, which:

1. Downloads the [SwissAdmin parallel corpus](https://dialektkarten.ch/static/swissadmin/swissadmin_aligned_2014_05_26.zip) (English–German, –French and –Italian). Its copyright lies with the Swiss government, so it is not part of this repository and is deleted again as soon as notebook 1 has finished.
2. Runs `synthetic_data_1_preprocess.ipynb`: splits the corpus into documents and sentence pairs, tokenises them into words, and removes boilerplate lines and every sentence that overlaps with the SwissGov-RSD test set.
3. Runs `synthetic_data_2_train_encoder.ipynb`: trains the word-level labeler, an ensemble of MLP heads on [KaLM-Embedding-Gemma3-12B](https://huggingface.co/tencent/KaLM-Embedding-Gemma3-12B-2511) word vectors, blended with a DiffAlign-style alignment score. It is trained on the projected iSTS-2016 data from the [SwissGov-RSD repository](https://github.com/ZurichNLP/SwissGov-RSD); the SwissGov-RSD validation split is only used to report how well the labeler works.
4. Runs `synthetic_data_3_create_data.ipynb`: labels every word of the corpus with the labeler (scores 5 = equivalent … 0 = unrelated, -1 = punctuation) and writes the chat-formatted training data for the fine-tuning.

### Start the run

If not already done, change to the `track_2a` directory.

```bash
cd track_2a/
```

```bash
make synthetic
```

Like `make run`, this creates `.env` from `.env.example` if it doesn't exist yet. The synthetic step only uses `HF_TOKEN`: set it if a Hugging Face download fails, for example because a model requires accepting its terms first.

The run takes several hours. The first run also downloads the ~24 GB encoder, which is kept in a Docker volume for later runs. The output of every cell is printed to the terminal as it runs.

### Results

All outputs are written next to the notebooks in `track_2a/src/synthetic_data/`:

- `executed/`: the three notebooks with all their outputs
- `pairs_en_{de,fr,it}*.json`, `overlap_*.json`, `removed_*.json`: the sentence pairs from notebook 1, and what was removed and why
- `rsd_head/labeler_ists.pt`: the trained labeler (the folder also holds several GB of cached encoder states)
- `synthetic/labels_en_{de,fr,it}.jsonl`: the word-level labels
- `synthetic/train_mixed.jsonl`, `synthetic/valid_mixed.jsonl`: the training data for the fine-tuning

These files contain text from the SwissAdmin corpus, so don't redistribute them.

## How to run the `train` notebook

### Prerequisites

- The same as for `synthetic_data`: Linux with an NVIDIA GPU, NVIDIA driver 580 or newer, Docker with the NVIDIA Container Toolkit
- The training data from `make synthetic` in `track_2a/src/synthetic_data/synthetic/` (`make train` stops with a message if it is missing)
- Access to [swiss-ai/Apertus-v1.5-8B](https://huggingface.co/swiss-ai/Apertus-v1.5-8B): the model is gated, so accept its terms on the model page and set `HF_TOKEN` in `.env`

### What it does

`make train` builds the container (the same image as `make synthetic`) and runs `synthetic_train_run_quantized.ipynb`, which:

1. Loads `train_mixed.jsonl` and `valid_mixed.jsonl` and drops examples longer than 12,288 tokens.
2. Rebalances the training set: it keeps every document in which at least 8% of the words are labelled as different, plus a random 30% of the others, so the model doesn't learn to answer "equivalent" everywhere.
3. Loads Apertus 1.5 8B in 4-bit and trains a QLoRA adapter on all linear layers of the language model, with a weighted loss: copied words count 0.1, scores 1.0, and scores below 5 three times as much.
4. Checks the result: sample outputs on the validation data, and Spearman correlation on the SwissGov-RSD validation split, both from the score probabilities and from real generated answers.

### Start the run

If not already done, change to the `track_2a` directory.

```bash
cd track_2a/
```

```bash
make train
```

`PILOT` in `.env` controls the size of the run:

- `PILOT=on` (default): a quick pilot on ~2,000 short documents for one epoch, to check that everything works
- `PILOT=off`: the full training (two epochs on the whole training set), as used for the published adapter. This takes many hours.

If the run is interrupted, `make train` resumes from the last checkpoint.

### Results

All outputs are written next to the notebook in `track_2a/src/train/`:

- `executed/`: the notebook with all its outputs, including the evaluation numbers
- `out/apertus-8b-rsd-mixed-weighted-pilot/` or `out/apertus-8b-rsd-mixed-weighted/` (full run): checkpoints, and in `final/` the adapter itself (`adapter_model.safetensors`, `adapter_config.json`, tokenizer and chat template). This is the same set of files as in [giandrin/apertus1.5-8b-rsd-mixed-weighted](https://huggingface.co/giandrin/apertus1.5-8b-rsd-mixed-weighted)
- `debug/`: the raw model output and target for one validation example

## How to run the `train_4gpus_clariden` notebook

Change `#SBATCH --account=<your_cscs_group_name>` in `train.sbatch`, copy all three files to Clariden and run train.sbatch.

The fully fine-tuned model can be accessed via [giandrin/apertus1.5-8b-rsd-mixed-weighted-full-fine-tune](https://huggingface.co/giandrin/apertus1.5-8b-rsd-mixed-weighted-full-fine-tune).
