# State-aware schema retrieval for agentic text-to-SQL

COMPSCI 646, Fall 2026, UMass Amherst.
Shalin, Sachin Mallick, Mauli Mehulkumar Patel, Manuj Jain.

## Files

- `schema_retrieval_baselines.py`: table retrieval baselines on pooled BIRD dev (BM25, dense BGE-base, dense + BGE reranker), with options for BIRD evidence and column descriptions.
- `results/`: outputs of our runs (`base.json`, `evid.json`, `desc.json`).
- `milestone1_final.tex`: milestone 1 report (ACM sigconf).

## Data

BIRD dev, release `dev_20240627`:

```
wget https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip
unzip dev.zip
unzip dev_20240627/dev_databases.zip -d dev_20240627/
```

## Run

```
pip install -r requirements.txt

python schema_retrieval_baselines.py --dev dev_20240627/dev.json --tables dev_20240627/dev_tables.json --setting pooled --methods bm25,dense,rerank --rerank_depth 20 --out results/base.json

python schema_retrieval_baselines.py --dev dev_20240627/dev.json --tables dev_20240627/dev_tables.json --setting pooled --methods dense,rerank --rerank_depth 20 --with_evidence --out results/evid.json

python schema_retrieval_baselines.py --dev dev_20240627/dev.json --tables dev_20240627/dev_tables.json --db_root dev_20240627/dev_databases --setting pooled --methods dense,rerank --rerank_depth 20 --out results/desc.json
```

Runs were done on a Colab T4 GPU.
