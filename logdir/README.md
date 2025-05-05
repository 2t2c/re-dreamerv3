# DreamerV3 Score Conversion and Plotting

The `/scores` directory contains the original results provided by the DreamerV3 paper. These results are **not** in the format required by the `plot.py`, however, so we need to convert them appropriately. After conversion they are stored in `/logdir/reproducibility/<task_name>/dreamerv3`. Reproducibility study results are stored in `/logdir/reproducibility/<task_name>/re_dreamerv3`.

## How to Convert Scores from `/scores` for `plot.py`

### 1. Extract the JSON file from `.gz`

Locate the scores you want in `/scores` and run the following command:

```bash
gzip -d <scores_you_want>.json.gz
```

### 2. Run the Score Conversion Script

Execute the following command to convert the JSON file to the format required by `plot.py`:

```bash
python json_score_converter.py \
  --input-file <extracted_scores_file_name>.json \
  --task-name <task_name_as_used_in_config_yaml> \
  --output_dir logdir/reproducibility
```

`plot.py` expects a specific file structure:

```
<task_name>/<method>/seed<seed>/scores.jsonl
```

so `json_score_converter.py` ensures this is the case.

## How to visualize
```bash
python plot.py
```