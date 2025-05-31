import wandb
import os
import argparse
import pandas as pd
from typing import List, Optional, Any, Dict, Set

def download_run_files(
    run_path: str, 
    output_dir: Optional[str] = None, 
    metrics_to_download_exact: Optional[List[str]] = None,
    metric_prefixes_to_download: Optional[List[str]] = None,
    download_media: bool = True
) -> None:
    """
    Downloads files and specified metrics (exact keys or by prefix) from a specific wandb run.

    Args:
        run_path (str): The path to the wandb run, in the format "entity/project/run_id".
        output_dir (str, optional): The directory to save the files and metrics to. 
                                     If None, a directory named after the run_id will be created.
        metrics_to_download_exact (List[str], optional): A list of exact metric keys to download.
        metric_prefixes_to_download (List[str], optional): A list of prefixes to find and download metrics.
        download_media (bool): Whether to download media files from the run. Defaults to True.
    """
    try:
        api = wandb.Api()
        run: wandb.apis.public.Run = api.run(run_path)

        if output_dir is None:
            effective_output_dir: str = run.id
        else:
            effective_output_dir: str = output_dir
        
        if not os.path.exists(effective_output_dir):
            os.makedirs(effective_output_dir)
            print(f"Created directory: {effective_output_dir}")

        # Download files
        if download_media:
            print(f"Downloading files for run: {run.name} ({run.id})")
            file_obj: wandb.apis.public.File
            for file_obj in run.files():
                print(f"Downloading {file_obj.name} to {effective_output_dir}...")
                file_obj.download(root=effective_output_dir, replace=True)
            print("File download complete.")
        else:
            print("Skipping media file download as per --no-media flag.")

        # Consolidate metrics to download
        final_metrics_to_download: Set[str] = set(metrics_to_download_exact) if metrics_to_download_exact else set()

        if metric_prefixes_to_download:
            print(f"Finding metrics with prefixes: {metric_prefixes_to_download}...")
            # Fetch all available metric keys from run.history_keys
            # This should be more comprehensive than run.summary.keys()
            try:
                history_keys_data = run.history_keys
                print(f"history_keys_data: {history_keys_data}")
                available_keys = history_keys_data.get('keys', {}).keys() if history_keys_data and 'keys' in history_keys_data else []
                # history_keys_data.get('keys') returns a dict like:
                # {'metric_name': {'type': 'number', 'min': 0, 'max': 1, 'mean': 0.5, ...}}
                # We just need the names, so .keys() on that dict.
                if not available_keys:
                     print("Warning: run.history_keys did not return any keys. Falling back to run.summary.keys().")
                     available_keys = list(run.summary.keys())
            except Exception as e:
                print(f"Warning: Could not fetch run.history_keys ({e}). Falling back to run.summary.keys().")
                available_keys = list(run.summary.keys())
            
            found_by_prefix_count = 0
            for prefix in metric_prefixes_to_download:
                for key in available_keys:
                    if key.startswith(prefix):
                        final_metrics_to_download.add(key)
                        found_by_prefix_count +=1
            print(f"Found {found_by_prefix_count} metrics matching specified prefixes.")

        if not final_metrics_to_download:
            print("No metrics specified or found by prefix for download.")
        else:
            print(f"Preparing to download metrics: {sorted(list(final_metrics_to_download))}")
            metric_key: str
            for metric_key in sorted(list(final_metrics_to_download)):
                print(f"Fetching metric: {metric_key}...")
                try:
                    df: pd.DataFrame = run.history(keys=[metric_key, '_step', '_timestamp'], pandas=True)

                    if df.empty or metric_key not in df.columns:
                        print(f"No data found for metric: {metric_key} or metric key not in history.")
                        continue

                    df = df[df[metric_key].notna()]

                    if df.empty:
                        print(f"No valid data points found for metric: {metric_key} after filtering.")
                        continue
                    
                    if '_timestamp' in df.columns:
                        df['walltime'] = pd.to_datetime(df['_timestamp'], unit='s')
                    else:
                        df['walltime'] = pd.NaT
                        print(f"Warning: '_timestamp' not found for metric {metric_key}. 'walltime' will be empty.")

                    desired_cols: List[str] = []
                    if '_step' in df.columns:
                        desired_cols.append('_step')
                    if 'walltime' in df.columns:
                        desired_cols.append('walltime')
                    if '_timestamp' in df.columns:
                        desired_cols.append('_timestamp')
                    desired_cols.append(metric_key)
                    
                    existing_desired_cols: List[str] = [col for col in desired_cols if col in df.columns]
                    other_cols: List[str] = [col for col in df.columns if col not in existing_desired_cols]
                    final_cols: List[str] = existing_desired_cols + other_cols
                    df = df[final_cols]
                    
                    if '_step' in df.columns:
                        df = df.rename(columns={'_step': 'step'})

                    # Create a filename by replacing slashes/spaces in the metric key
                    safe_metric_filename = metric_key.replace(os.sep, '_').replace('/', '_').replace(' ', '_') + '.csv'
                    
                    # Save directly into the effective_output_dir (no subdirectories based on metric key)
                    output_path_for_metric = os.path.join(effective_output_dir, safe_metric_filename)

                    # No need to create subdirectories for the metric file itself here
                    # as we are saving flatly in effective_output_dir.
                    # output_metric_dir = os.path.dirname(output_path_for_metric)
                    # if not os.path.exists(output_metric_dir):
                    #     os.makedirs(output_metric_dir)
                    #     print(f"Created metric subdirectory: {output_metric_dir}")

                    df.to_csv(output_path_for_metric, index=False)
                    print(f"Saved metric {metric_key} to {output_path_for_metric}")
                except Exception as e:
                    print(f"Could not download metric {metric_key}: {e}")
            print("Metrics download complete.")
        
        print("All requested data downloaded successfully.")

    except wandb.errors.CommError as e:
        print(f"A wandb communication error occurred: {e}")
        print("Please ensure you are logged into wandb (run 'wandb login'), the run path is correct, and your network connection is stable.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("Please ensure pandas is installed ('pip install pandas').")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download files and metrics (exact or by prefix) from a wandb run.")
    parser.add_argument("run_path", type=str, help="Path to the wandb run (e.g., 'entity/project/run_id')")
    parser.add_argument("--output_dir", type=str, default=None, help="Base directory to save files and metrics (optional)")
    parser.add_argument("--metrics", nargs='*', default=[], dest='metrics_to_download_exact', help="List of exact metric keys to download (e.g., 'train/loss/total' 'val/accuracy')")
    parser.add_argument("--metric-prefixes", nargs='*', default=[], dest='metric_prefixes_to_download', help="List of prefixes to find and download metrics (e.g., 'train/loss' 'charts')")
    parser.add_argument("--no-media", action='store_false', dest='download_media', help="Set this flag to skip downloading media files.")
    parser.set_defaults(download_media=True)

    args: argparse.Namespace = parser.parse_args()

    download_run_files(
        args.run_path, 
        args.output_dir, 
        args.metrics_to_download_exact, 
        args.metric_prefixes_to_download, 
        args.download_media
    )