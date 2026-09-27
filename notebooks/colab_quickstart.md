# Colab quickstart (two cells)

## Cell 1 — bootstrap (install, mount Drive, validate, init local runtime)

```python
!pip install -q "git+https://github.com/betaanoiar1-gif/tradek.git@arena/01a0e49f-tradek"

from trading_school_ai.colab import bootstrap

info = bootstrap(runtime_dir="/content/tsa_runtime")
print("drive mounted:", info["drive_mounted"])
print("runtime:", info["runtime_dir"], "| db:", info["db_path"])
print("dataset:", info["canonical_path"])
print("doctor ok:", info["doctor_ok"])
for c in info["doctor"]["checks"]:
    if not c["ok"]:
        print(" ", c["severity"].upper(), c["check"], "-", c["detail"])
```

If `doctor_ok` is False because the dataset is missing, set the path explicitly:

```python
import os
os.environ["TSA_CANONICAL_PATH"] = "/content/drive/MyDrive/.../BTCUSDT-1m-canonical.parquet"
os.environ["TSA_GAPS_PATH"] = "/content/drive/MyDrive/.../BTCUSDT-1m-missing-gaps.parquet"
```

Nothing is downloaded or synthesized to replace a missing dataset.

## Cell 2 — run doctor + an AI_OFF learning cycle

```python
# `pip install` ships the package, not the repository's YAML files.
# Generate a real config first (AI_OFF and paper execution by default):
!tsa init-config -o /content/tsa.yaml --runtime /content/tsa_runtime

!tsa doctor          -c /content/tsa.yaml
!tsa learn --ai-off  -c /content/tsa.yaml
!tsa report --markdown -c /content/tsa.yaml
!tsa sync            -c /content/tsa.yaml   # checksum-verified backup to Drive
```

`pip install` targets only this package and its declared dependencies; it does not
force-upgrade unrelated Colab packages.
