# Embedding clustering for raw security events (prototype)

This repository contains a proof of concept (PoC) that converts raw, pre-normalization security telemetry (one event per line) into transformer embeddings, performs unsupervised clustering, and produces artifacts for exploratory analysis and visualization.

The main script is `embed_to_tb.py`.


## What it does

Given an input file with events, the pipeline:

1. Loads events (default mode expects a plain text file: one line equals one event).
2. Canonicalizes high-entropy fragments (timestamps, hashes, numbers, and optionally IPs) to reduce false diversity.
3. Computes sentence-level embeddings using a pre-trained transformer encoder (`sentence-transformers/all-MiniLM-L6-v2` by default).
4. Clusters embeddings (HDBSCAN by default, with an optional k-means baseline).
5. Writes artifacts for TensorBoard Embedding Projector and for quick human review (cluster statistics and examples).
6. Optionally computes a 2D UMAP projection and saves an interactive HTML scatter plot.


## Requirements

- Python 3.10+
- Recommended: a virtual environment

Dependencies:

- `sentence-transformers`
- `numpy`, `pandas`
- `hdbscan` (for the default clustering method)
- `umap-learn` and `plotly` (optional, for UMAP and interactive plots)
- `scikit-learn` (optional, for k-means baseline)

Example installation:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip

pip install sentence-transformers numpy pandas hdbscan
pip install umap-learn plotly scikit-learn
```


## Dataset used in experiments

In experiments, an open dataset from the Scalyr Samples repository was used. The dataset was formed by concatenating multiple text log files into a single file, where each line is treated as an event.

Reference dataset repository:

- Scalyr Samples logs: https://github.com/scalyr/samples/tree/main/logs


## Quick start

### 1) Prepare an input file

Create a plain text file where each line is a single event:

```text
[dhcp] DNS Update Request ... client=10.0.0.1
[dhcp] DNS Update Successful ... client=10.0.0.1
...
```

### 2) Run the pipeline (HDBSCAN clustering)

```bash
python embed_to_tb.py --input ./your_events.txt --format text --out tb_projector --umap
```

The default model is `sentence-transformers/all-MiniLM-L6-v2`, and the default clustering method is HDBSCAN.


## Output artifacts

The script writes outputs into `--out` (default: `tb_projector/`):

- `vectors.tsv` - normalized embeddings (one row per event).
- `metadata.tsv` - event metadata for TensorBoard Projector. Includes:
  - `raw` (original line),
  - `norm` (canonicalized line),
  - `len` (length of canonicalized line),
  - `cluster` (assigned cluster label).
- `projector_config.pbtxt` - a minimal config for TensorBoard Projector.
- `cluster_summary.csv` - cluster ranking by size and share.
- `cluster_examples.csv` - example `raw` and `norm` lines per cluster.
- `events_with_clusters.csv` - full table of events plus `cluster` and optional UMAP coordinates.
- `clusters.html` (optional) - interactive UMAP scatter plot (requires `--umap`).
- `cluster_trends.html` (optional) - per-cluster volume over time, if timestamps can be extracted.


## Canonicalization controls

Canonicalization reduces false diversity from high-cardinality tokens. Be careful with flow-like logs.

Key flags:

- `--no-mask-ips`  
  Keep IPv4 addresses intact. Useful when clustering by source or destination IP.

- `--numbers-min-len N`  
  Mask only numbers with at least N digits. For flow logs, set `N=6` to preserve ports (<= 5 digits).

- `--no-mask-numbers`  
  Disable number masking entirely.

Example (keep IPs, preserve ports):

```bash
python embed_to_tb.py --input flows.txt --format text --no-mask-ips --numbers-min-len 6 --out tb_projector
```


## TensorBoard Embedding Projector

Run TensorBoard and open the Projector UI:

```bash
tensorboard --logdir tb_projector --port 6006
```

Then open `http://localhost:6006` in a browser and go to the Projector tab. Load `vectors.tsv` and `metadata.tsv` if needed. The script also writes `projector_config.pbtxt`.


## Notes on metrics

Embeddings are L2-normalized (`normalize_embeddings=True`). For unit-length vectors, Euclidean distance is monotonically related to cosine distance, so using `metric="euclidean"` in HDBSCAN is consistent with cosine-based similarity interpretation.


## Reproducibility and limitations

- This is a PoC oriented to offline, batch analysis.
- The current implementation does not build an ANN index. For interactive production scenarios, consider adding FAISS or a similar index.
- Results depend on canonicalization rules and clustering hyperparameters. Validate any telemetry policy changes with security requirements in mind.


## License

This project is licensed under the Apache License, Version 2.0 (Apache-2.0). See the `LICENSE` file for details.

## How to cite

If you use this idea or the prototype in an academic or technical publication, you can cite it using the BibTeX entry below.

### Preprint / manuscript

```bibtex
@misc{Tumakov2025NoisePatternMiningPreprint,
  author       = {Tumakov, Maksim},
  title        = {Noise Pattern Mining in Unnormalized Security Events Using Transformer Embeddings and Clustering},
  year         = {2025},
  url          = {https://github.com/mt-labx/macro2micro}
}
