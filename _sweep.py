import pickle
import time
from pathlib import Path

import numpy as np

import q5_v3
from q5_identify import load_patterns

patterns = load_patterns(q5_v3.ROOT / "patterns")
scenes = q5_v3.labelled_scenes()
store = Path("cache_v3/proposals")
store.mkdir(exist_ok=True)
base = dict(q5_v3.PARAMS)

cache = {}
for tag, folder, truth, true_name in scenes:
    path = store / f"{tag}.pkl"
    if path.exists():
        cache[tag] = pickle.loads(path.read_bytes())
        continue
    t = time.time()
    q5_v3.prepare([(tag, folder, truth, true_name)], patterns, base, verbose=False, proposal_cache=cache)
    path.write_bytes(pickle.dumps(cache[tag]))
    print(f"proposals {tag} {time.time()-t:.0f}s", flush=True)

configs = [dict(rule=r, select=s, high=0.90) for r in ("excess", "blend") for s in ("raw", "residual")]
configs += [dict(rule="excess", select=s, high=h) for s in ("raw", "residual") for h in (0.85, 0.93)]
results = []
for cfg in configs:
    t = time.time()
    p = dict(base, **cfg)
    prep = q5_v3.prepare(scenes, patterns, p, verbose=False, proposal_cache=cache)
    ids = {tag: d["name"] == d["true_name"] for tag, d in prep.items()}
    real = [v for k, v in ids.items() if k.startswith("train_")]
    syn = [v for k, v in ids.items() if k.startswith("syn_")]
    results.append((cfg, sum(real), sum(syn), prep))
    print(f"{cfg} real-ID {sum(real)}/{len(real)} synthetic-ID {sum(syn)}/{len(syn)} {time.time()-t:.0f}s", flush=True)

best = max(results, key=lambda r: r[1] + r[2])
print("best", best[0], flush=True)
for tag, d in best[3].items():
    rank = next((k for k, r in enumerate(d["table"]) if r[1] == d["true_name"]), None)
    print(f"  {tag:14} true {d['true_name']:16} pred {d['name']:16} rank {rank}", flush=True)
rows = q5_v3.score_prepared(best[3], dict(base, **best[0]), verbose=True)
real = [m["total"] for k, m in rows if k.startswith("train_")]
syn = [m["total"] for k, m in rows if k.startswith("syn_")]
print(f"mean total real {np.mean(real):.3f} synthetic {np.mean(syn):.3f}", flush=True)
Path("cache_v3/best_config.pkl").write_bytes(pickle.dumps(best[0]))
