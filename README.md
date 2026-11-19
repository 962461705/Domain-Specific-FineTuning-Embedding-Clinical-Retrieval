# Domain-Specific Fine-Tuning of Embedding Models for Clinical Specialty Knowledge-Enhanced Retrieval

Code and disease ontologies for the paper:

> "Improving Medical Dense Retrieval via Ontology-Guided Hard-Negative Learning," submitted to *IEEE Access* (under review).

We study hard-negative mining for Chinese medical question–answer retrieval in two disease domains, liver disease and neurological disease. A lightweight disease ontology for each domain labels the diseases named in questions and answers. The labels are used to

1. measure how many mined hard negatives name the same disease as the question (likely false negatives),
2. remove same-disease and unnamed candidates from BM25 or dense candidate lists before a hard negative is chosen,
3. build negative pools that need no retrieval model,
4. inject random same-disease negatives in a controlled dose experiment, and
5. rerank retrieved answers at inference time.

The scripts reproduce all experiments, tables, and figures of the paper. **No data or model weights are included in this repository** (see [Data](#data)).

---

## Repository structure

```
liver/                              Liver-disease domain (main domain)
  data-clean/
    Coarse_screening_liver.py       Step A: keyword filter (domain lexicon)
    Fine_screening_liver.py         Step B: LLM filter (DeepSeek-R1-Distill-Qwen-32B)
  data-preprocess/
    0_data_splicing.py              00  concatenate the filtered files of each source
    01_data_convert.py              01  convert and clean one source (run once per source)
    02_data_merge.py                02  merge sources, remove cross-source exact duplicates
    03_data_cluster.py              03  near-duplicate clustering (MinHash LSH)
    04_data_spilit.py               04  leakage-controlled train/dev/test split
    05_data_figures.py              05  data tables and figures for the paper
  train/
    06_ontology annotation_new.py   06  disease ontology and labels
    07_get_neg_new.py               07  BM25 and dense candidate mining (top-100)
    08_training_sets_new.py         08  one training file per negative-selection method
    09_train_eval_new.py            09  fine-tuning, test evaluation, significance tests
    10_rerank_new.py                10  inference-time ontology reranking
    11_paper_figures_new.py         11  result tables and figures (also cross-domain tables)

neuro/                              Neurological-disease domain (replication domain)
  data-clean/                       Coarse_screening_neuro.py, Fine_screening_neuro.py
  data-pre-process/                 0_data_splicing_neuro.py ... 05_data_figures_neuro.py
  train/                            06_ontology_annotation_neuro_new.py ... 11_paper_figures_neuro_new.py
```

The two domains use the same pipeline; the neurological scripts differ only in the lexicon, the ontology, the paths, and the models used (BGE-base-zh only).

---

## Requirements

- Linux, Python 3.10, CUDA 12.4
- GPUs: one GPU per fine-tuning run (we used NVIDIA L20, 48 GB, up to four runs in parallel). The LLM filter in `Fine_screening_*.py` used four GPUs with tensor parallelism.
- Python packages (the versions we used):

```
torch==2.6.0
sentence-transformers==5.1.2
transformers==4.51.3
datasets==4.4.1
vllm==0.8.5                          # only for Fine_screening_*.py
jieba==0.42.1
datasketch==2.0.0
opencc-python-reimplemented==0.1.7
numpy==2.2.6
scipy==1.15.3
tqdm==4.67.1
matplotlib==3.10.7                   # only for 05 and 11
python-docx==1.2.0                   # only for 05 and 11
```

Save the list above as `requirements.txt` and install it with `pip install -r requirements.txt` (remove the comments after `#` if your pip version complains). Other versions may also work but have not been tested.

---

## Data

The question–answer data are **not redistributed** here. Please download the original datasets from their releases and follow their licenses:

| Source | Link |
|---|---|
| CMDD (Chinese Medical Dialogue Data) | https://github.com/Toyhom/Chinese-medical-dialogue-data |
| Huatuo-26M | https://github.com/FreedomIntelligence/Huatuo-26M |
| MedChatZH | https://github.com/tyang816/MedChatZH |

Models (download from Hugging Face):

| Model | Used in |
|---|---|
| `BAAI/bge-base-zh-v1.5` | 07, 09, 10 (both domains) |
| `Qwen/Qwen3-Embedding-0.6B` | 07, 09, 10 (liver domain) |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | Fine_screening_*.py |

---

## Paths: please edit before running

The scripts contain **absolute paths of our server**, for example

```python
OUT_DIR = '/home/liver/data/liver_dataset'
'/home/liver/data/HuaTuo/clean_liver_train_datasets.jsonl'
'path': '/data/bge-base-zh-v1.5'
```

Before running, replace them with your own paths. They are near the top of each script:

| Script | What to edit |
|---|---|
| `Coarse_screening_*.py` | `input_filename`, `output_filename` at the end of the file |
| `Fine_screening_*.py` | `MODEL_PATH`, `INPUT_FILE`, `OUTPUT_FILE`, `FILTERED_FILE` |
| `0_data_splicing*.py` | `OUT_DIR` and the file lists in `SOURCES` |
| `01`–`05`, `11` | `OUT_DIR` (must be the same in all scripts of a domain); in the liver `05_data_figures.py` and `11_paper_figures_new.py`, also `NEURO_DIR`, the output directory of the neurological domain used for the cross-domain tables |
| `06`–`10` | output directory via `--out-dir` or the environment variable `LIVER_OUT_DIR` / `NEURO_OUT_DIR`; model paths in the `'path'` entries of `07`, `09`, `10`; fine-tuned models via `--model-root` |

All intermediate files and reports are written under `OUT_DIR`.

---

## How to run

The commands below are for the liver domain; replace the script names for the neurological domain.

### 1. Domain filtering (data-clean)

Run on each file of each source (CMDD, Huatuo-26M, MedChatZH). Edit the input and output paths in the scripts for every file.

```bash
python Coarse_screening_liver.py      # keyword filter
python Fine_screening_liver.py        # LLM filter; outputs *_clean.jsonl
```

The `*_clean.jsonl` files are the inputs of step 00.

### 2. Data processing (data-preprocess)

```bash
python 0_data_splicing.py
python 01_data_convert.py             # set SOURCE_NAME = 'cmdd', then 'huatuo26m', then 'medchatzh'
python 02_data_merge.py
python 03_data_cluster.py
python 04_data_spilit.py
python 05_data_figures.py             # optional: data tables and figures
```

`01_data_convert.py` processes one source at a time: change `SOURCE_NAME` at the top of the file and run it three times.

### 3. Ontology, candidates, and training sets

```bash
python "06_ontology annotation_new.py"
python 07_get_neg_new.py --model bge-base       # also builds the shared BM25 candidates
python 07_get_neg_new.py --model qwen3-0.6b
python 08_training_sets_new.py --suite main     # main experiment
python 08_training_sets_new.py --suite core     # ablation and BM25/dense + ontology filter
python 08_training_sets_new.py --suite dose     # same-disease dose experiment
```

### 4. Fine-tuning and evaluation

Each suite of step 08 is trained as a separate set of runs (`--tag`). `--launch` runs all methods × seeds in parallel on the listed GPUs.

```bash
python 09_train_eval_new.py --launch --tag new      --models bge-base,qwen3-0.6b --gpus 0,1,2,3
python 09_train_eval_new.py --launch --tag new_core --models bge-base,qwen3-0.6b --gpus 0,1,2,3
python 09_train_eval_new.py --launch --tag new_dose --models bge-base,qwen3-0.6b --gpus 0,1,2,3

python 09_train_eval_new.py --report --tag new      # results and paired tests
python 09_train_eval_new.py --report --tag new_core
python 09_train_eval_new.py --report --tag new_dose
```

A quick check before the full runs:

```bash
python 09_train_eval_new.py --model bge-base --method onto_pool_rand --seed 42 --max-steps 50
```

### 5. Ontology reranking

```bash
python 10_rerank_new.py --models bge-base,qwen3-0.6b --methods mnrl,random,onto_pool_rand,margin --seeds 42,43,44
```

### 6. Tables and figures

```bash
python 11_paper_figures_new.py
```

The liver scripts `05_data_figures.py` and `11_paper_figures_new.py` also read the neurological outputs (`NEURO_DIR`) and write the cross-domain tables; run them after both domains are finished.

---

## Method names in the code and in the paper

| Code | Paper |
|---|---|
| `mnrl` | In-batch only (MNRL) |
| `bm25`, `dense` | BM25 top-1, Dense top-1 |
| `dense_rank10`, `dense_rank100` | Dense rank 10, Dense rank 100 |
| `random` | Random |
| `margin` | TopK-PercPos |
| `margin_onto` | TopK-PercPos + ontology filter |
| `bm25_onto`, `dense_onto` | BM25 / Dense top-1 + ontology filter |
| `onto_pool_rand`, `onto_pool_strat` | Ontology (safe) pool, random / family-stratified |
| `onto_loose_rand`, `onto_strat_loose` | Loose pool, random / family-stratified |
| `onto_dose_p10`, `p30`, `p50` | Ontology pool + 10% / 30% / 50% same-disease |
| `onto_pool_sim`, `onto_pool_hard`, `onto_hard_loose`, `onto_pool_mid`, `onto_nmid_loose`, `onto_hmid_loose` | Ablation variants (Appendix B of the paper) |

---

## Notes

- The disease ontologies (concepts, Chinese synonyms, disease families, and disease–entity relations) are defined inside `06_ontology annotation_new.py` and `06_ontology_annotation_neuro_new.py`, with the source of every relation. They were compiled from ICD-10, Chinese Wikipedia, Wikidata, the MSD Manual (Chinese consumer edition), and MedlinePlus, and **have not been reviewed by clinicians**.
- The answers in the datasets come from online sources and have not been checked for clinical accuracy. The code and fine-tuned models are for research only and **must not be used for medical advice or clinical decisions**.
- Results may differ slightly from the paper because of hardware, library versions, and GPU nondeterminism.

---

## Citation

The paper is currently under review. Citation details will be added here once it is published.

## License

The code in this repository is released under the MIT License (see `LICENSE`). The datasets, models, and ontology sources keep their own licenses.
