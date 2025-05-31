import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import os
import re
from typing import Optional, List, Set

COLORS = [
    '#0022ff', '#33aa00', '#ff0011', '#ddaa00', '#cc44dd', '#0088aa',
    '#001177', '#117700', '#990022', '#885500', '#553366', '#006666',
    '#7777cc', '#999999', '#990099', '#888800', '#ff00aa', '#444444',
]

def natfmt(x):
    if abs(x) < 1e3:
        val, suffix = x, ''
    elif 1e3 <= abs(x) < 1e6:
        val, suffix = x / 1e3, 'K'
    elif 1e6 <= abs(x) < 1e9:
        val, suffix = x / 1e6, 'M'
    elif 1e9 <= abs(x):
        val, suffix = x / 1e9, 'B'
    else:
        val, suffix = x, ''
    
    if abs(val) == 0:
        return '0'
    if abs(val) < 10 and not isinstance(val, int):
        return f'{val:.1f}{suffix}'
    return f'{val:.0f}{suffix}'

def style_plot_ax(ax, xticks=6, yticks=8, grid_min_subdivisions=1):
    ax.tick_params(axis='x', which='major', length=3, labelsize=10, pad=3)
    ax.tick_params(axis='y', which='major', length=3, labelsize=10, pad=2)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(xticks))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(yticks))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: natfmt(x)))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: natfmt(x)))
    grid_color = '#eeeeee'
    ax.grid(which='major', color=grid_color, linestyle='-', linewidth=0.7)
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(grid_min_subdivisions + 1))
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(grid_min_subdivisions + 1))
    ax.grid(which='minor', color=grid_color, linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', length=0)

def apply_legend_style_and_adjust_layout(fig, ax, num_legend_entries):
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        fig.tight_layout()
        return

    legend_options = dict(
        fontsize=10, numpoints=1, labelspacing=0.2, columnspacing=1.2,
        handlelength=1.5, handletextpad=0.5, loc='lower center', frameon=True,
        ncol=min(num_legend_entries, 4)
    )

    leg = fig.legend(handles, labels, **legend_options)
    if leg:
        leg.get_frame().set_edgecolor('#dddddd')
        leg.set_zorder(2000)
        for linehandle in leg.legend_handles:
            linehandle.set_linewidth(2)
        
        fig.canvas.draw()
        extent = leg.get_window_extent(fig.canvas.get_renderer())
        extent = extent.transformed(fig.transFigure.inverted())
        
        fig.tight_layout(rect=[0, extent.y1 + 0.01, 1, 0.95], h_pad=1.0, w_pad=1.0)
    else:
        fig.tight_layout()

def plot_metrics_from_csv(
    task: str, 
    envs: List[str], 
    losses_to_plot_param: List[str],
    output_dir_path: Optional[str] = None
) -> None:
    """
    For each specified loss (or all discoverable losses if none specified),
    reads CSV files from different environments and plots them together on a single graph.
    Saves one plot per loss type.

    Args:
        task (str): The main task identifier, used for structuring CSV input paths.
        envs (List[str]): A list of environment names.
        losses_to_plot_param (List[str]): A list of loss keys to plot. If empty, discovers all.
        output_dir_path (str, optional): Directory to save the generated plots. 
                                         If None, defaults to 'wandb_exports/plots/{task}/'.
    """
    base_csv_dir_template = f"wandb_exports/losses/{task}/"

    actual_losses_to_plot: List[str]

    if not losses_to_plot_param:
        print("No specific losses provided. Discovering available metrics...")
        discovered_metric_keys: Set[str] = set()
        for env_name in envs:
            env_metric_dir = os.path.join(base_csv_dir_template, env_name)
            if os.path.isdir(env_metric_dir):
                for filename in os.listdir(env_metric_dir):
                    if filename.endswith(".csv"):
                        metric_key_repr = filename[:-4] 
                        discovered_metric_keys.add(metric_key_repr)
            else:
                print(f"Info: Directory not found for discovery: {env_metric_dir}")
        
        if not discovered_metric_keys:
            print(f"No CSV metrics found for task '{task}' in any of the specified environments: {envs}. Nothing to plot.")
            return
        actual_losses_to_plot = sorted(list(discovered_metric_keys))
        print(f"Discovered metrics to plot: {actual_losses_to_plot}")
    else:
        actual_losses_to_plot = losses_to_plot_param

    if not actual_losses_to_plot:
        print("No metrics to plot. Exiting.")
        return

    if output_dir_path is None:
        effective_output_plot_dir = f"wandb_exports/plots/{task}/"
    else:
        effective_output_plot_dir = output_dir_path

    if not os.path.exists(effective_output_plot_dir):
        os.makedirs(effective_output_plot_dir)
        print(f"Created plot directory: {effective_output_plot_dir}")

    for loss_key_repr in actual_losses_to_plot:
        fig, ax = plt.subplots(figsize=(12, 7.5))
        style_plot_ax(ax, xticks=6, yticks=8, grid_min_subdivisions=2)

        plot_title = f"Metric: {loss_key_repr} for Task: {task}"
        ax.set_title(plot_title, fontsize=16)
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel(loss_key_repr, fontsize=12)

        found_data_for_this_loss = False
        color_index = 0
        num_plotted_envs = 0

        for env_name in envs:
            csv_filename = loss_key_repr + ".csv"
            csv_path = os.path.join(base_csv_dir_template, env_name, csv_filename)

            if not os.path.exists(csv_path):
                print(f"Info: CSV file not found at {csv_path} for env '{env_name}', metric '{loss_key_repr}'. Skipping.")
                continue

            try:
                df = pd.read_csv(csv_path)
                if 'step' not in df.columns:
                    print(f"Warning: 'step' column not found in {csv_path}. Skipping for env '{env_name}'.")
                    continue
                
                potential_metric_columns = [col for col in df.columns if col.lower() not in ['step', 'walltime', '_timestamp']]
                if not potential_metric_columns:
                    print(f"Warning: No suitable data columns found in {csv_path} for env '{env_name}'. Columns: {df.columns.tolist()}. Skipping.")
                    continue
                metric_column_to_plot = potential_metric_columns[-1]
                print(f"Info: For file '{csv_filename}', plotting column '{metric_column_to_plot}' for env '{env_name}'.")

                line_color = COLORS[color_index % len(COLORS)]
                color_index += 1
                ax.plot(df['step'], df[metric_column_to_plot], linestyle='-', label=f"{env_name}", color=line_color, linewidth=1.5)
                found_data_for_this_loss = True
                num_plotted_envs +=1
            except Exception as e:
                print(f"An error occurred while processing {csv_path} for metric '{loss_key_repr}', env '{env_name}': {e}")

        if found_data_for_this_loss:
            apply_legend_style_and_adjust_layout(fig, ax, num_plotted_envs)
            plot_filename = task + "_" + loss_key_repr + ".png"
            final_plot_path = os.path.join(effective_output_plot_dir, plot_filename)
            plt.savefig(final_plot_path, dpi=150)
            print(f"Plot for '{loss_key_repr}' saved to {final_plot_path}")
        else:
            print(f"No data plotted for metric '{loss_key_repr}'. Plot not saved.")
            fig.tight_layout()
        
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot metrics from CSV files. For each specified loss (or all discoverable if none specified), generates a plot combining data from all specified environments.")
    parser.add_argument("--task", default="dmc_walker_run", type=str, help="Task name (e.g., 'dreamerv3_antmaze') used to find CSVs under 'wandb_exports/losses/TASK_NAME/'.")
    parser.add_argument("--envs", default=["reproduce", "rssmv2"], nargs='+', help="List of environment names (subdirectories under 'wandb_exports/losses/TASK_NAME/') to plot (e.g., 'medium' 'umaze').")
    parser.add_argument("--losses", default=[], nargs='*', help="List of loss keys (transformed, e.g., 'train_loss_image') to plot. If empty, discovers all available .csv files as metrics.")
    parser.add_argument("--output_dir", default="wandb_exports/plots", type=str, help="Directory to save the generated plots. Optional. Defaults to 'wandb_exports/plots/TASK_NAME/' if not specified, or current dir if task is also default.")

    args = parser.parse_args()
    plot_metrics_from_csv(args.task, args.envs, args.losses, args.output_dir)