# Mushroom Collector (Windows / VS Code)

## Setup (Windows / VS Code)
1) Create venv
- PowerShell:
  - `py -m venv .venv`
  - `\.\.venv\Scripts\Activate.ps1`

2) Install deps
- `pip install -r requirements.txt`

3) Run
- Local output:
  - `python run.py --output-dir .\mushroom_data`

- With sharding (example shard 0 of 5):
  - PowerShell:
    - `$env:N_SHARDS="5"; $env:SHARD_ID="0"; $env:PARALLEL_WORKERS="4"; python run.py --output-dir .\mushroom_data`

- Disable progress bars:
  - `python run.py --no-progress-bar --output-dir .\mushroom_data`

## Notes
- Your original Colab uninstall/install blocks are not needed on Windows; use a clean venv instead.
- GCS output requires installing `google-cloud-storage` and setting `--gcs-bucket` or `GCS_BUCKET`.
