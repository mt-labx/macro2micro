import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# pip install umap-learn hdbscan matplotlib
import umap
import hdbscan
import matplotlib.pyplot as plt


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UMAP + clustering visualization for event embeddings")
    p.add_argument("--vectors", default="tb_projector/vectors.tsv", help="Path to vectors.tsv")
    p.add_argument("--meta", default="tb_projector/metadata.tsv", help="Path to metadata.tsv")
    p.add_argument("--out_dir", default="figures", help="Output directory")
    p.add_argument("--out_prefix", default="umap_clusters", help="Output filename prefix")

    # Performance controls
    p.add_argument("--max_points", type=int, default=40000, help="Max points to plot (subsample for speed)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    # Clustering (HDBSCAN)
    p.add_argument("--min_cluster_size", type=int, default=50)
    p.add_argument("--min_samples", type=int, default=10)

    # UMAP (visualization only)
    p.add_argument("--n_neighbors", type=int, default=15)
    p.add_argument("--min_dist", type=float, default=0.05)

    # Plot controls
    p.add_argument("--top_k", type=int, default=12, help="Plot only top-K clusters with distinct colors")
    p.add_argument("--show_legend", action="store_true", help="Show legend (recommended only when clusters <= ~10)")

    return p.parse_args()


def stratified_subsample(df: pd.DataFrame, label_col: str, max_points: int, seed: int) -> pd.DataFrame:
    """Subsample rows while keeping representation across clusters and noise."""
    if len(df) <= max_points:
        return df

    rng = np.random.default_rng(seed)

    # Ensure at least a small quota per cluster, then fill the rest proportionally
    labels = df[label_col].to_numpy()
    unique, counts = np.unique(labels, return_counts=True)

    # Base quota per label
    base = max(5, int(0.02 * max_points / max(1, len(unique))))

    idx_selected = []
    remaining_budget = max_points

    for lab, cnt in sorted(zip(unique, counts), key=lambda x: x[1], reverse=True):
        take = min(cnt, base)
        rows = df.index[df[label_col] == lab].to_numpy()
        chosen = rng.choice(rows, size=take, replace=False)
        idx_selected.append(chosen)
        remaining_budget -= take
        if remaining_budget <= 0:
            break

    idx_selected = np.concatenate(idx_selected) if idx_selected else np.array([], dtype=int)

    if remaining_budget > 0:
        not_taken = df.index.difference(idx_selected)
        if len(not_taken) > 0:
            extra = rng.choice(not_taken.to_numpy(), size=min(remaining_budget, len(not_taken)), replace=False)
            idx_selected = np.concatenate([idx_selected, extra])

    return df.loc[idx_selected]


def main() -> None:
    args = parse_args()

    vectors_path = Path(args.vectors)
    meta_path = Path(args.meta)

    if not vectors_path.exists():
        raise FileNotFoundError(f"Vectors file not found: {vectors_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load
    X = np.loadtxt(vectors_path, delimiter="\t")
    meta = pd.read_csv(meta_path, sep="\t")

    if len(meta) != X.shape[0]:
        raise ValueError(f"Row mismatch: vectors={X.shape[0]} meta={len(meta)}")

    # 2) Normalize vectors so cosine and euclidean are aligned
    Xn = l2_normalize(X)

    # 3) Cluster in embedding space
    # Note: we use euclidean on L2-normalized vectors (monotonic w.r.t cosine distance)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(Xn)
    meta["cluster_id"] = labels

    # 4) UMAP projection for visualization only
    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric="cosine",
        random_state=args.seed,
    )
    proj = reducer.fit_transform(Xn)
    meta["umap_x"] = proj[:, 0]
    meta["umap_y"] = proj[:, 1]

    # 5) Rank clusters by size (exclude noise = -1)
    counts = meta["cluster_id"].value_counts()
    noise_n = int(counts.get(-1, 0))
    cluster_counts = counts.drop(index=-1, errors="ignore")
    top_clusters = list(cluster_counts.head(args.top_k).index)

    # 6) Prepare plotting dataframe (subsample for speed)
    plot_df = stratified_subsample(meta, label_col="cluster_id", max_points=args.max_points, seed=args.seed)

    # 7) Plot
    fig, ax = plt.subplots(figsize=(7, 5))

    # Plot noise first
    noise = plot_df["cluster_id"] == -1
    if noise.any():
        ax.scatter(
            plot_df.loc[noise, "umap_x"],
            plot_df.loc[noise, "umap_y"],
            s=3,
            alpha=0.12,
            label=f"Noise (-1), n={int(noise.sum())}",
            rasterized=True,
        )

    # Plot top clusters in default matplotlib color cycle
    for i, cid in enumerate(top_clusters):
        m = plot_df["cluster_id"] == cid
        if not m.any():
            continue
        ax.scatter(
            plot_df.loc[m, "umap_x"],
            plot_df.loc[m, "umap_y"],
            s=3,
            alpha=0.25,
            label=f"Cluster {cid}, n={int(m.sum())}",
            rasterized=True,
        )

    # Plot the rest as "Other" if present
    other = (~noise) & (~plot_df["cluster_id"].isin(top_clusters))
    if other.any():
        ax.scatter(
            plot_df.loc[other, "umap_x"],
            plot_df.loc[other, "umap_y"],
            s=3,
            alpha=0.10,
            label=f"Other clusters, n={int(other.sum())}",
            rasterized=True,
        )

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

    # Keep the figure clean for papers: prefer caption in LaTeX over title here
    # ax.set_title("UMAP projection of embeddings")

    if args.show_legend and (len(top_clusters) <= 10):
        ax.legend(markerscale=3, fontsize=6, ncol=2, frameon=False)

    fig.tight_layout()

    pdf_path = out_dir / f"{args.out_prefix}.pdf"
    png_path = out_dir / f"{args.out_prefix}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=600)

    # Save enriched metadata for reproducibility
    meta_out = out_dir / f"{args.out_prefix}__metadata_with_clusters.tsv"
    meta.to_csv(meta_out, sep="\t", index=False)

    # Small console summary
    n_total = len(meta)
    n_clusters = int((cluster_counts > 0).sum())
    print(f"Total events: {n_total}")
    print(f"Clusters (excluding noise): {n_clusters}")
    print(f"Noise (-1): {noise_n} ({noise_n / max(1, n_total):.2%})")
    if len(cluster_counts) > 0:
        top_k_cov = cluster_counts.head(args.top_k).sum() / n_total
        print(f"Top-{args.top_k} coverage: {top_k_cov:.2%}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    print(f"Saved: {meta_out}")


if __name__ == "__main__":
    main()