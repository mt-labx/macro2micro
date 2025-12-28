import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

try:
    import umap
except Exception:  # pragma: no cover
    umap = None

try:
    import hdbscan
except Exception:  # pragma: no cover
    hdbscan = None

try:
    import plotly.express as px
except Exception:  # pragma: no cover
    px = None

# Defaults tuned for a fast PoC on a laptop.
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LIMIT = 100000
DEFAULT_OUT_DIR = "tb_projector"

_STATUS_CLASS_WORD = {
    "1": "ONE",
    "2": "TWO",
    "3": "THREE",
    "4": "FOUR",
    "5": "FIVE",
}


# -------------------------
# Normalization utilities
# -------------------------


@dataclass(frozen=True)
class NormConfig:
    # High-entropy token masking toggles
    mask_hex: bool = True
    mask_ts: bool = True
    mask_ips: bool = True
    mask_numbers: bool = True
    # Mask numbers only if they have at least this many digits.
    # For flow-like logs, set to 6 so ports (<=5 digits) are preserved.
    numbers_min_len: int = 1


def normalize_text(s: str, cfg: NormConfig) -> str:
    """Canonicalize text so embeddings are less sensitive to volatile tokens.

    Important: if you mask *all* numbers and IPs on flow-style logs, many lines become
    effectively identical and embeddings may collapse to the same vector.
    Use --numbers-min-len and --no-mask-ips to preserve discriminative tokens.
    """
    s = str(s).strip()

    # Preserve status class BEFORE number masking.
    def _status_repl(m: re.Match) -> str:
        code = m.group(1)
        first = code[0]
        word = _STATUS_CLASS_WORD.get(first, "X")
        return f"status=STATUS_{word}XX"

    s = re.sub(r"\bstatus=(\d{3})\b", _status_repl, s)

    if cfg.mask_hex:
        # hashes/uuid-like
        s = re.sub(r"\b[0-9a-fA-F]{8,}\b", "<HEX>", s)

    if cfg.mask_ts:
        # ISO-ish timestamps
        s = re.sub(
            r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?Z?\b",
            "<TS>",
            s,
        )

    ip_pat = r"\b\d{1,3}(?:\.\d{1,3}){3}\b"

    # IPv4 handling:
    # - if mask_ips=True: replace IPs with <IP>
    # - if mask_ips=False: keep IPs intact AND protect them from number masking
    ip_tokens: Dict[str, str] = {}

    def _ip_key(i: int) -> str:
        # Create a key with NO digits so it won't be affected by number masking.
        a = chr(65 + (i % 26))
        b = chr(65 + ((i // 26) % 26))
        c = chr(65 + ((i // 676) % 26))
        return f"__IP{c}{b}{a}__"

    if cfg.mask_ips:
        s = re.sub(ip_pat, "<IP>", s)
    else:
        def _protect_ip(m: re.Match) -> str:
            k = _ip_key(len(ip_tokens))
            ip_tokens[k] = m.group(0)
            return k

        s = re.sub(ip_pat, _protect_ip, s)

    if cfg.mask_numbers:
        n = max(1, int(cfg.numbers_min_len))
        s = re.sub(rf"\b\d{{{n},}}\b", "<N>", s)

    # Restore protected IP literals (if any)
    if ip_tokens:
        for k, v in ip_tokens.items():
            s = s.replace(k, v)

    return s[:2000]


def tsv_safe(v: object) -> str:
    s = "" if pd.isna(v) else str(v)
    return s.replace("\t", " ").replace("\r", " ").replace("\n", " ")


# -------------------------
# Input loading
# -------------------------

def detect_format(path: Path, forced: str) -> str:
    f = forced.strip().lower()
    if f and f != "auto":
        return f

    ext = path.suffix.lower()
    if ext == ".csv":
        return "csv"
    if ext in (".jsonl", ".ndjson"):
        return "jsonl"
    return "text"


def load_csv(path: Path, limit: int, delimiter: str, encoding: str, no_header: bool) -> pd.DataFrame:
    df = pd.read_csv(path, sep=delimiter, encoding=encoding, header=None if no_header else "infer")
    if no_header:
        df.columns = [f"col{i}" for i in range(df.shape[1])]
    if df.empty:
        raise ValueError("Input CSV is empty")
    return df.head(limit).copy()


def load_jsonl(path: Path, limit: int, encoding: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding=encoding, errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                # If a line is not valid JSON, treat it as a message.
                rows.append({"message": line})
            if len(rows) >= limit:
                break

    if not rows:
        raise ValueError("Input JSONL is empty")
    return pd.DataFrame(rows)


_SOURCE_PREFIX_RE = re.compile(r"^\[(?P<source>[^\]]{1,64})\]\s+")
_SOURCE_KV_RE = re.compile(r"\bsource=(?P<source>[A-Za-z0-9_.-]{1,64})\b")


def load_text(path: Path, limit: int, encoding: str) -> pd.DataFrame:
    messages: List[str] = []
    sources: List[str] = []
    line_no: List[int] = []

    with open(path, "r", encoding=encoding, errors="ignore") as f:
        for i, line in enumerate(f, 1):
            s = line.strip("\n")
            if not s.strip():
                continue

            src = ""
            m1 = _SOURCE_PREFIX_RE.match(s)
            if m1:
                src = m1.group("source")
                s = s[m1.end():]
            else:
                m2 = _SOURCE_KV_RE.search(s)
                if m2:
                    src = m2.group("source")

            messages.append(s)
            sources.append(src)
            line_no.append(i)

            if len(messages) >= limit:
                break

    if not messages:
        raise ValueError("Input text file is empty")

    return pd.DataFrame({"line_no": line_no, "source": sources, "message": messages})



def pick_first_existing(cols: List[str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None


def row_to_text_generic(row: pd.Series, text_col: Optional[str]) -> str:
    """Generic row-to-text for unknown schemas.

    Priority:
    1) explicit --text-col
    2) common message-like columns
    3) compact key=value rendering of a subset of columns
    """
    if text_col and text_col in row.index:
        return str(row[text_col])

    idx = list(row.index)
    msg = pick_first_existing(idx, ["message", "msg", "log", "event", "line", "request"])  # broad
    if msg and pd.notna(row[msg]) and str(row[msg]).strip():
        return str(row[msg])

    # Compact key=value fallback (cap number of fields)
    kv: List[str] = []
    for c in idx[:30]:
        v = row.get(c)
        if pd.isna(v):
            continue
        sv = str(v).strip()
        if not sv:
            continue
        kv.append(f"{c}={sv[:200]}")
    return " ".join(kv)



# -------------------------
# Metadata / timestamps
# -------------------------

def choose_timestamp_series(
    df: pd.DataFrame,
    fmt: str,
    explicit_ts_col: str,
    json_ts_keys: List[str],
    raw_text: List[str],
) -> Optional[pd.Series]:
    """Try to find a timestamp for trend charts.

    Priority:
    1) explicit --ts-col
    2) common dataframe columns (date, timestamp, @timestamp, time)
    3) json ts keys
    4) regex extraction from raw text (ISO only)
    """
    if explicit_ts_col and explicit_ts_col in df.columns:
        ts = pd.to_datetime(df[explicit_ts_col], errors="coerce")
        return ts

    for c in ["date", "@timestamp", "timestamp", "time", "ts"]:
        if c in df.columns:
            ts = pd.to_datetime(df[c], errors="coerce")
            if ts.notna().any():
                return ts

    if fmt == "jsonl":
        for k in json_ts_keys:
            if k in df.columns:
                ts = pd.to_datetime(df[k], errors="coerce")
                if ts.notna().any():
                    return ts

    # Last resort: try to extract ISO-ish timestamps from raw text
    ser = pd.Series(raw_text)
    iso = ser.str.extract(
        r"(?P<ts>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)",
        expand=False,
    )
    if iso.notna().any():
        ts = pd.to_datetime(iso, errors="coerce")
        if ts.notna().any():
            return ts

    return None


def build_metadata(
    df: pd.DataFrame,
    raw_text: List[str],
    norm_text: List[str],
    meta_cols: List[str],
    keep_all_meta: bool,
) -> pd.DataFrame:
    """Create a metadata table for Projector and CSV exports."""
    if keep_all_meta:
        base = df.copy()
        # Avoid huge wide tables if JSON has many keys.
        if base.shape[1] > 80:
            base = base.iloc[:, :80].copy()
    else:
        # Default set + user-defined columns
        preferred = [
            "date",
            "@timestamp",
            "timestamp",
            "time",
            "ts",
            "source",
            "logtype",
            "service",
            "app",
            "host",
            "hostname",
            "remote_host",
            "request",
            "status",
            "sent",
            "referer",
            "user_agent",
            "level",
            "severity",
            "message",
            "line_no",
        ]
        preferred = preferred + [c for c in meta_cols if c]
        keep = [c for c in preferred if c in df.columns]
        base = df[keep].copy() if keep else pd.DataFrame(index=df.index)

    base = base.copy()
    base.insert(0, "raw", [tsv_safe(x) for x in raw_text])
    base.insert(1, "norm", [tsv_safe(x) for x in norm_text])
    base.insert(2, "len", [len(x) for x in norm_text])

    # Sanitize all string-like columns
    for c in base.columns:
        if c == "len":
            continue
        base[c] = base[c].map(tsv_safe)

    return base


# -------------------------
# Clustering / visualization
# -------------------------

def ensure_plotly_available() -> None:
    if px is None:
        raise RuntimeError("plotly is not installed. Install with: pip install plotly")


def ensure_umap_available() -> None:
    if umap is None:
        raise RuntimeError("umap-learn is not installed. Install with: pip install umap-learn")


def cluster_embeddings(
    emb: np.ndarray,
    method: str,
    n_clusters: int,
    min_cluster_size: int,
    min_samples: int,
) -> np.ndarray:
    """Return cluster labels for each embedding."""
    method = method.lower().strip()
    if method in ("none", "no", "off"):
        return np.full((emb.shape[0],), -1, dtype=int)

    if method == "hdbscan":
        if hdbscan is None:
            raise RuntimeError("hdbscan is not installed. Install with: pip install hdbscan")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",  # unit-normalized embeddings => euclidean ~ cosine monotonic
        )
        return clusterer.fit_predict(emb)

    if method == "kmeans":
        from sklearn.cluster import MiniBatchKMeans

        km = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            batch_size=4096,
            n_init="auto",
        )
        return km.fit_predict(emb)

    raise ValueError(f"Unknown clustering method: {method}")


# -------------------------
# CLI
# -------------------------

def parse_args() -> argparse.Namespace:

    p = argparse.ArgumentParser(
        description=(
            "PoC: logs -> transformer embeddings -> clustering -> Projector TSV + reports. "
            "Supports CSV, JSONL/NDJSON, and plain text (one event per line)."
        )
    )

    p.add_argument("--input", type=str, required=True, help="Path to input file")
    p.add_argument(
        "--format",
        type=str,
        default="auto",
        help="Input format: auto | csv | jsonl | text",
    )
    p.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Max number of events")
    p.add_argument("--encoding", type=str, default="utf-8", help="File encoding")

    # CSV
    p.add_argument("--csv-delim", type=str, default=",", help="CSV delimiter")
    p.add_argument(
        "--csv-no-header",
        action="store_true",
        help="Treat CSV as having no header row (header=None).",
    )

    # Normalization controls (important for flow-style numeric logs)
    p.add_argument(
        "--no-mask-ips",
        action="store_true",
        help="Do not mask IPv4 addresses (<IP>).",
    )
    p.add_argument(
        "--numbers-min-len",
        type=int,
        default=1,
        help=(
            "Mask numbers only if they have at least this many digits. "
            "For flow logs, set to 6 to preserve ports (<=5 digits)."
        ),
    )
    p.add_argument(
        "--no-mask-numbers",
        action="store_true",
        help="Do not mask numbers at all.",
    )

    # Text selection
    p.add_argument(
        "--text-col",
        type=str,
        default="",
        help=(
            "For CSV/JSONL: column/key to embed (e.g. request/message). "
            "If omitted, the script will construct text automatically."
        ),
    )

    # Metadata controls
    p.add_argument(
        "--meta-cols",
        type=str,
        default="",
        help="Comma-separated list of additional columns to include in metadata.tsv",
    )
    p.add_argument(
        "--keep-all-meta",
        action="store_true",
        help="Include many/all columns in metadata (capped to first ~80 columns)",
    )

    # Timestamp controls for trend charts
    p.add_argument(
        "--ts-col",
        type=str,
        default="",
        help="Column/key to use as timestamp for trend charts (optional)",
    )
    p.add_argument(
        "--json-ts-keys",
        type=str,
        default="@timestamp,timestamp,time,date,ts",
        help="Comma-separated JSONL timestamp keys to try (when --ts-col is not set)",
    )

    # Embeddings
    p.add_argument("--model", type=str, default=DEFAULT_MODEL, help="SentenceTransformer model")
    p.add_argument("--out", type=str, default=DEFAULT_OUT_DIR, help="Output directory")

    # Clustering
    p.add_argument(
        "--cluster",
        type=str,
        default="hdbscan",
        help="Clustering method: hdbscan | kmeans | none",
    )
    p.add_argument("--n-clusters", type=int, default=30, help="For kmeans: number of clusters")
    p.add_argument("--min-cluster-size", type=int, default=50, help="For hdbscan: minimum cluster size")
    p.add_argument("--min-samples", type=int, default=10, help="For hdbscan: min_samples")

    # Visualization
    p.add_argument(
        "--umap",
        action="store_true",
        help="Compute 2D UMAP projection and save interactive HTML scatter",
    )
    p.add_argument(
        "--topk-trend",
        type=int,
        default=8,
        help="If timestamps exist: plot trends for top-K clusters",
    )
    p.add_argument(
        "--time-bin",
        type=str,
        default="1H",
        help="Time bucket for trend plot (e.g., 5min, 15min, 1H, 1D)",
    )

    return p.parse_args()


# -------------------------
# Main
# -------------------------

def main() -> None:
    args = parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise FileNotFoundError(f"Input file not found: {inp}")

    fmt = detect_format(inp, args.format)

    if fmt == "csv":
        df = load_csv(inp, limit=args.limit, delimiter=args.csv_delim, encoding=args.encoding, no_header=bool(args.csv_no_header))
    elif fmt == "jsonl":
        df = load_jsonl(inp, limit=args.limit, encoding=args.encoding)
    elif fmt == "text":
        df = load_text(inp, limit=args.limit, encoding=args.encoding)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    if df.empty:
        raise ValueError("No events loaded")

    text_col = args.text_col.strip() or None

    raw_text = [row_to_text_generic(df.iloc[i], text_col) for i in range(len(df))]

    cfg = NormConfig(
        mask_ips=not bool(args.no_mask_ips),
        mask_numbers=not bool(args.no_mask_numbers),
        numbers_min_len=int(args.numbers_min_len),
    )
    norm_text = [normalize_text(x, cfg) for x in raw_text]

    model = SentenceTransformer(args.model)
    emb = model.encode(
        norm_text,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    emb = np.asarray(emb)

    out_dir = Path(args.out)
    os.makedirs(out_dir, exist_ok=True)

    vectors_path = out_dir / "vectors.tsv"
    metadata_path = out_dir / "metadata.tsv"

    # 1) Write vectors
    with open(vectors_path, "w", encoding="utf-8") as vf:
        for v in emb:
            vf.write("\t".join(f"{x:.6f}" for x in v) + "\n")

    # 2) Clustering
    labels = cluster_embeddings(
        emb=emb,
        method=args.cluster,
        n_clusters=args.n_clusters,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
    )
    cluster_str = ["noise" if int(x) == -1 else f"c{int(x)}" for x in labels]

    # 3) Metadata
    meta_cols = [c.strip() for c in args.meta_cols.split(",") if c.strip()]
    json_ts_keys = [c.strip() for c in args.json_ts_keys.split(",") if c.strip()]

    meta = build_metadata(
        df=df,
        raw_text=raw_text,
        norm_text=norm_text,
        meta_cols=meta_cols,
        keep_all_meta=bool(args.keep_all_meta),
    )
    meta["cluster"] = cluster_str

    # Save metadata TSV for TensorBoard Projector
    meta.to_csv(metadata_path, sep="\t", index=False)

    # 4) Summaries
    summary = (
        pd.Series(cluster_str, name="cluster")
        .value_counts(dropna=False)
        .rename_axis("cluster")
        .reset_index(name="count")
    )
    summary["share"] = summary["count"] / len(meta)
    summary_path = out_dir / "cluster_summary.csv"
    summary.to_csv(summary_path, index=False)

    # A few examples per cluster
    examples = (
        meta[["cluster", "raw", "norm"]]
        .groupby("cluster", as_index=False)
        .head(30)
        .reset_index(drop=True)
    )
    examples_path = out_dir / "cluster_examples.csv"
    examples.to_csv(examples_path, index=False)

    events_path = out_dir / "events_with_clusters.csv"
    meta.to_csv(events_path, index=False)

    # 5) Optional: 2D UMAP scatter
    scatter_path = None
    if args.umap:
        ensure_umap_available()
        ensure_plotly_available()

        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.05,
            metric="cosine",
            random_state=42,
        )
        z = reducer.fit_transform(emb)
        meta["umap_x"] = z[:, 0]
        meta["umap_y"] = z[:, 1]
        meta.to_csv(events_path, index=False)

        hover_cols = [c for c in meta.columns if c not in ("umap_x", "umap_y")]
        # Limit hover columns for performance
        hover_cols = hover_cols[:12]

        fig = px.scatter(
            meta,
            x="umap_x",
            y="umap_y",
            color="cluster",
            hover_data=hover_cols,
            render_mode="webgl",
            title="Embedding clusters (UMAP 2D)",
        )
        scatter_path = out_dir / "clusters.html"
        fig.write_html(scatter_path, include_plotlyjs="cdn")

    # 6) Optional: trends by cluster if timestamp can be found
    trend_path = None
    ts = choose_timestamp_series(
        df=df,
        fmt=fmt,
        explicit_ts_col=args.ts_col.strip(),
        json_ts_keys=json_ts_keys,
        raw_text=raw_text,
    )

    if ts is not None and ts.notna().any():
        try:
            ensure_plotly_available()
            tmp = pd.DataFrame({"ts": ts, "cluster": cluster_str}).dropna(subset=["ts"]).copy()
            tmp["bucket"] = tmp["ts"].dt.floor(args.time_bin)

            top = (
                pd.Series(cluster_str)
                .value_counts()
                .head(max(1, args.topk_trend))
                .index
                .tolist()
            )
            tmp = tmp[tmp["cluster"].isin(top)]

            agg = tmp.groupby(["bucket", "cluster"], as_index=False).size()
            fig2 = px.line(
                agg,
                x="bucket",
                y="size",
                color="cluster",
                title=f"Cluster volume over time (bucket={args.time_bin})",
            )
            trend_path = out_dir / "cluster_trends.html"
            fig2.write_html(trend_path, include_plotlyjs="cdn")
        except Exception:
            trend_path = None

    # 7) TensorBoard Projector config
    config_path = out_dir / "projector_config.pbtxt"
    config_path.write_text(
        """
embeddings {
  tensor_path: "vectors.tsv"
  metadata_path: "metadata.tsv"
}
""".lstrip(),
        encoding="utf-8",
    )

    print("Input:")
    print(" -", str(inp))
    print(" - format:", fmt)
    print(" - events:", len(df))

    print("\nWrote:")
    for p in [vectors_path, metadata_path, config_path, summary_path, examples_path, events_path]:
        print(" -", str(p))
    if scatter_path:
        print(" -", str(scatter_path))
    if trend_path:
        print(" -", str(trend_path))

    print("\nRun TensorBoard:")
    print(f"  tensorboard --logdir {out_dir} --port 6006")


if __name__ == "__main__":
    main()