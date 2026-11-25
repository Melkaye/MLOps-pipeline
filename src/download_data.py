import os
import yaml

from ucimlrepo import fetch_ucirepo

cfg = yaml.safe_load(open("configs/config.yaml"))["data"]

heart = fetch_ucirepo(id=cfg["uci_id"])
df = heart.data.features.copy()
df[cfg["target"]] = (heart.data.targets["num"] > 0).astype(int)

os.makedirs(os.path.dirname(cfg["path"]), exist_ok=True)
df.to_csv(cfg["path"], index=False)
print(f"Saved {df.shape} to {cfg['path']}")
