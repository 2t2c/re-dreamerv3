import json
import argparse
from pathlib import Path
import sys 

def convert_json_to_jsonl(input_filepath, target_task, output_basedir, env_name):
    """
    Converts data for a specific task from a structured JSON file
    to JSON Lines format, creating separate output files per run
    in directories named '<env>-<task>-<method>-seed<seed>'.

    Args:
        input_filepath (Path): Path to the input JSON file.
        target_task (str): The specific task name to extract data for.
        output_basedir (Path): The base directory to create run-specific folders in.
        env_name (str): Name of the environment.
    """
    print(f"Loading input file: {input_filepath}")
    try:
        with open(input_filepath, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_filepath}")
        sys.exit(1) # Exit if input not found
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_filepath}. Is it a valid JSON file?")
        sys.exit(1) # Exit on invalid JSON
    except Exception as e:
        print(f"An unexpected error occurred while reading {input_filepath}: {e}")
        sys.exit(1) # Exit on other read errors

    # --- Input Validation ---
    if not isinstance(data, list):
        print(f"Error: Expected input file '{input_filepath}' to contain a JSON list ([...]). Found type: {type(data)}")
        sys.exit(1) # Exit if format is wrong

    processed_count = 0
    records_found_for_task = 0
    # --- Process Each Record in the Input List ---
    for i, record in enumerate(data):
        record_index_for_logging = i + 1 # Use 1-based indexing for logs
        if not isinstance(record, dict):
            print(f"Warning: Skipping item #{record_index_for_logging} in the input list as it's not a dictionary.")
            continue

        # Check for required keys (including 'method' now)
        required_keys = {'task', 'method', 'seed', 'xs', 'ys'}
        if not required_keys.issubset(record.keys()):
            missing = required_keys - record.keys()
            print(f"Warning: Skipping item #{record_index_for_logging}. Missing required keys: {missing}. Found keys: {list(record.keys())}")
            continue

        record_task = record['task']
        record_method = record['method'] # Extract method
        record_seed = record['seed']
        xs = record['xs']
        ys = record['ys']

        # --- Task Filtering ---
        if record_task != target_task:
            # This record is not for the target task, move to the next one
            continue

        # Found a record for the target task
        records_found_for_task += 1
        print(f"Processing record #{record_index_for_logging}: Task='{record_task}', Method='{record_method}', Seed={record_seed}")

        # --- Data Integrity Check ---
        if not isinstance(xs, list) or not isinstance(ys, list):
            print(f"  Error: For record #{record_index_for_logging}, 'xs' or 'ys' is not a list. Skipping.")
            continue
        if len(xs) != len(ys):
            print(f"  Error: For record #{record_index_for_logging}, 'xs' (len {len(xs)}) and 'ys' (len {len(ys)}) have different lengths. Skipping.")
            continue
        if not xs:
            print(f"  Warning: For record #{record_index_for_logging}, 'xs' and 'ys' lists are empty. Skipping.")
            continue

        # --- Prepare Output ---
        # Construct the new directory name
        run_dir_name = f"{env_name}-{record_task}-{record_method}-seed{record_seed}"
        run_dir = output_basedir / run_dir_name

        try:
            run_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"  Error: Could not create output directory {run_dir}: {e}")
            continue # Skip this run if directory creation fails

        output_filepath = run_dir / "scores.jsonl"

        # --- Write JSON Lines Output ---
        print(f"  Writing output to: {output_filepath}")
        lines_written = 0
        try:
            with open(output_filepath, 'w') as outfile:
                for step, score in zip(xs, ys):
                    # Create the target dictionary format
                    output_record = {"step": step, "episode/score": score}
                    # Dump as JSON string and add newline
                    json_line = json.dumps(output_record)
                    outfile.write(json_line + '\n')
                    lines_written += 1
            print(f"  Successfully wrote {lines_written} lines.")
            processed_count += 1
        except IOError as e:
            print(f"  Error: Could not write to output file {output_filepath}: {e}")
        except Exception as e:
            print(f"  An unexpected error occurred while writing {output_filepath}: {e}")

    # Final summary
    print("-" * 30)
    if records_found_for_task == 0:
        print(f"Warning: No records found matching the target task '{target_task}' in {input_filepath}.")
    elif processed_count == 0:
         print(f"Found {records_found_for_task} record(s) for task '{target_task}', but encountered errors during processing or data validation. No files were successfully written.")
    else:
        print(f"Finished processing.")
        print(f"Found {records_found_for_task} record(s) matching task '{target_task}'.")
        print(f"Successfully converted and wrote data for {processed_count} run(s).")
    print("-" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Convert structured JSON run data to JSON Lines format for a specific task. "
            "Output is organized into directories named '<env>-<task>-<method>-seed<seed>'."
        )
    )
    parser.add_argument(
        "-i", "--input-file",
        type=str,
        required=True,
        help="Path to the input JSON file (list-of-dictionaries format)."
    )
    parser.add_argument(
        "-e", "--env-name",
        type=str,
        required=True,
        help="The environment name used in the names of the output files."
    )
    parser.add_argument(
        "-t", "--task-name",
        type=str,
        required=True,
        help="The exact task name to extract data for (used for filtering input records)."
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="converted_runs",
        help="Base directory where run-specific folders will be created."
    )

    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_base_path = Path(args.output_dir)

    convert_json_to_jsonl(input_path, args.task_name, output_base_path, args.env_name)