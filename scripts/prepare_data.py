"""
Generate MATCC train/valid/test datasets from LOCAL Qlib data for one market.

Usage:
    conda run -n matcc python scripts/prepare_data.py --universe csi300 --tag 2009_2025
    conda run -n matcc python scripts/prepare_data.py --universe sp500  --tag 2009_2025
    conda run -n matcc python scripts/prepare_data.py --universe csi300 --smoke

It reads util/{universe}[_smoke].yaml, builds the MASTERTSDatasetH (Alpha158 158-dim
stock feats + 63-dim market feats) and writes one pickle per split to
dataset/{universe}/{universe}_dl_{split}_{tag}.pkl.

Notes:
  * No network: local Qlib data is assumed present (no GetData().qlib_data download).
  * qlib loads ".py" module_paths relative to CWD, so every ".py" module_path in the
    config is rewritten to an absolute path under util/ before instantiation.
  * Market-index feature data is verified to exist before any heavy compute.
"""

import argparse
import hashlib
import json
import os
import pickle
import sys

# Make `src.baseline_utils` importable when run from anywhere.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import yaml

from qlib.constant import REG_CN, REG_US
from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import init_instance_by_config

from src.baseline_utils import UTIL_DIR, dataset_path, ensure_parent, yaml_path


REGION_CONST = {"cn": REG_CN, "us": REG_US}


def config_hash(cfg):
    """Hash the complete normalized YAML configuration.

    The fingerprint covers provider, region, handlers, processors, expressions,
    segments, market indices and dataset parameters.  Any configuration change must
    invalidate both the fitted-handler cache and the prepared split files.
    """
    blob = json.dumps(cfg, sort_keys=True, default=str, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def dataset_manifest_path(universe, tag):
    return os.path.join(ROOT, "dataset", universe, f"{universe}_dataset_{tag}.json")


def dataset_cache_matches(universe, tag, fingerprint):
    """Return True only when every split and its matching manifest are present."""
    if not all(os.path.isfile(dataset_path(universe, tag, split))
               for split in ("train", "valid", "test")):
        return False
    try:
        with open(dataset_manifest_path(universe, tag), "r") as f:
            manifest = json.load(f)
    except (OSError, ValueError, TypeError):
        return False
    return (
        manifest.get("version") == 1
        and manifest.get("universe") == universe
        and manifest.get("tag") == tag
        and manifest.get("config_hash") == fingerprint
    )


def write_dataset_manifest(universe, tag, fingerprint):
    """Atomically publish the manifest after all three splits are safely written."""
    path = dataset_manifest_path(universe, tag)
    ensure_parent(path)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({
            "version": 1,
            "universe": universe,
            "tag": tag,
            "config_hash": fingerprint,
            "splits": ["train", "valid", "test"],
        }, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def normalize_module_paths(obj):
    """Convert '.py' module_paths to dotted 'util.<name>' so classes are picklable.

    qlib's get_module_by_module_path loads '.py' paths via spec_from_file_location,
    which derives a garbled module name -> the fitted handler cache then fails to pickle
    (PicklingError). A dotted module path makes qlib use importlib.import_module instead,
    giving the classes a stable, importable '__module__' (ROOT is on sys.path, so
    'util.DropExtremeLabel' / 'util.MATCC_dataset' resolve).
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "module_path" and isinstance(v, str) and v.endswith(".py"):
                obj[k] = "util." + v[:-3]  # "DropExtremeLabel.py" -> "util.DropExtremeLabel"
            else:
                normalize_module_paths(v)
    elif isinstance(obj, list):
        for item in obj:
            normalize_module_paths(item)
    return obj


def check_market_indices(cfg):
    provider = os.path.expanduser(cfg["qlib_init"]["provider_uri"])
    indices = cfg["market_data_handler_config"]["market_indices"]
    missing = [i for i in indices if not os.path.isdir(os.path.join(provider, "features", i))]
    if missing:
        raise SystemExit(
            f"Missing local feature data for market indices {missing} under "
            f"{provider}/features/. For CN the repo substitutes CSI1000 (sh000852) for "
            f"the unavailable CSI100 (sh000903); check util/{cfg['task']}.yaml."
        )


def _handler_cache_loadable(path):
    try:
        with open(path, "rb") as f:
            pickle.load(f)
        return True
    except Exception:
        return False


def build_dataset(cfg, universe, tag):
    """Instantiate the qlib dataset, with a cached fitted handler.

    The handler cache is validated before reuse (a killed/partial earlier write would
    otherwise leave a corrupt pickle) and written atomically (tmp + rename).
    """
    handler_conf = cfg["task"]["dataset"]["kwargs"]["handler"]
    seg = cfg["task"]["dataset"]["kwargs"]["segments"]
    h_cache = os.path.join(
        UTIL_DIR,
        f"handler_{universe}_{tag}_{str(seg['train'][0])[:10]}_{str(seg['test'][1])[:10]}"
        f"_{config_hash(cfg)}.pkl")

    if os.path.exists(h_cache) and not _handler_cache_loadable(h_cache):
        print(f"[prepare_data] removing corrupt handler cache: {h_cache}")
        os.remove(h_cache)

    if os.path.exists(h_cache):
        print(f"[prepare_data] reusing cached handler: {h_cache}")
    else:
        print("[prepare_data] building & fitting Alpha158 handler (this may take a while)...")
        h = init_instance_by_config(handler_conf)
        tmp = h_cache + ".tmp"
        h.to_pickle(tmp, dump_all=True)
        os.replace(tmp, h_cache)  # atomic
        print(f"[prepare_data] cached handler -> {h_cache}")

    cfg["task"]["dataset"]["kwargs"]["handler"] = f"file://{h_cache}"
    print("[prepare_data] instantiating MASTERTSDatasetH ...")
    return init_instance_by_config(cfg["task"]["dataset"])


def report(split, ds):
    try:
        idx = ds.get_index()
    except Exception:
        idx = None
    if idx is None or len(idx) == 0:
        print(f"[prepare_data]   {split}: EMPTY (no samples)")
        return
    dates = idx.get_level_values("datetime")
    try:
        sample_shape = tuple(ds[0].shape)
    except Exception as e:
        sample_shape = f"<unreadable: {e}>"
    print(f"[prepare_data]   {split}: N={len(idx)} "
          f"date {dates.min().date()}..{dates.max().date()} sample_shape={sample_shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", required=True, choices=["csi300", "sp500"])
    ap.add_argument("--tag", default="2009_2025")
    ap.add_argument("--smoke", action="store_true", help="use the tiny-window *_smoke.yaml")
    ap.add_argument("--force", action="store_true", help="rebuild even if all splits exist")
    args = ap.parse_args()

    tag = "smoke" if args.smoke else args.tag
    ypath = yaml_path(args.universe, smoke=args.smoke)
    with open(ypath, "r") as f:
        cfg = yaml.safe_load(f)

    normalize_module_paths(cfg)
    fingerprint = config_hash(cfg)
    if not args.force and dataset_cache_matches(args.universe, tag, fingerprint):
        print(f"[prepare_data] dataset cache matches config {fingerprint} for "
              f"{args.universe}/{tag}; skipping (use --force to rebuild).")
        return
    if not args.force:
        print(f"[prepare_data] dataset cache missing or stale for {args.universe}/{tag}; "
              f"rebuilding with config {fingerprint}.")
    check_market_indices(cfg)

    region_str = cfg["qlib_init"]["region"]
    provider_uri = cfg["qlib_init"]["provider_uri"]
    print(f"[prepare_data] universe={args.universe} tag={tag} region={region_str} "
          f"provider={provider_uri}")

    import qlib
    qlib.init(provider_uri=provider_uri, region=REGION_CONST[region_str])

    dataset = build_dataset(cfg, args.universe, tag)

    print("[prepare_data] preparing splits ...")
    for split in ("train", "valid", "test"):
        ds = dataset.prepare(split, col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        out = dataset_path(args.universe, tag, split)
        ensure_parent(out)
        tmp = out + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(ds, f)
        os.replace(tmp, out)
        report(split, ds)
        print(f"[prepare_data]   -> {out}")

    write_dataset_manifest(args.universe, tag, fingerprint)
    print(f"[prepare_data] manifest -> {dataset_manifest_path(args.universe, tag)}")
    print("[prepare_data] DONE.")


if __name__ == "__main__":
    main()
