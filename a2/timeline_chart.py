"""
To save time this file was mostly written by Google's Gemini CLI. The quality of the code produced by the agent is
impressive in my opinion. Only minor adjustments in naming and organizing the code have to be made to receive usable
plotting utility functions.

The file hosts two methods. The create_gantt_chart method holds all burger orders on the x-axis and displays the entire
simulation run as a timeline. Here we can observe how long an order waits until it gets processed. It is important to
remember that this plot shows the process from the burger order perspective. prep_wait time does not suggest that the
prep station has to wait all the time.

To view the process at Theke 3 from the resource perspective (i.e. the workers) the method create_resource_chart is more
useful. The plot shows the utilization of the linecooks as well as the utilization of the burgermaster. Here it becomes
clear that the burgermaster has high downtime and often waits for the preparation of the warm ingredients. This also
visually confirms the numerical bottleneck that has been identified in the analysis step of the burgenerator.

To view the fill levels of the containers over time the method create_container_chart has been implemented. It shows
resource depletion and refill over the entire simulation horizon for each container and the different scenarios.

The charts for the deviation of the baseline in respect to average wait time of the burgermaster as well as the average
leftover ingredients are created as barplots showing the percentage change of the scenario in comparison to the baseline
scenario.
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import a2.config as c
from matplotlib.patches import Patch

from a2 import burgenerator_storage

C_MAP = {
    'prep_wait': '#afe1f5',
    'prep_wait_for_refill': '#69a0cb',
    'prep_work': '#3a609c',
    'assembly_wait_to_begin': '#ffd700',
    'assembly_wait_for_refill': '#ffb14e',
    'assembly_wait': '#d1a7ef',
    'assembly_work': '#ba55aa',
    'assembly_packing': '#840853'
}

C_MAP_THRESHOLD = {
    'Baseline': '#840853',
    'Safety Stock 10.0%': '#d1a7ef',
    'Safety Stock 12.5%': '#ba55aa',
    'Safety Stock 17.5%': '#afe1f5',
    'Safety Stock 20.0%': '#69a0cb',
}

C_MAP_CAPACITY = {
    'Baseline': '#840853',
    'Container Size 60': '#afe1f5',
    'Container Size 90': '#69a0cb',
}


def create_gantt_chart(timeline_events: pd.DataFrame, path):
    """Creates and saves a Gantt chart of the burger simulation."""
    timeline_events['duration'] = timeline_events['end'] - timeline_events['start']

    timeline_events.dropna(inplace=True)

    fig, ax = plt.subplots(figsize=(20, 10))

    y_labels = sorted(timeline_events['order_id'].unique(), reverse=True)
    y_pos = np.arange(len(y_labels))

    ax.set_yticks(y_pos[13::10])
    ax.set_yticklabels(y_labels[13::10])
    ax.set_ylim(min(y_pos) - 1, max(y_pos) + 1)

    for index, event in timeline_events.iterrows():
        order_idx = y_labels.index(event['order_id'])
        ax.barh(order_idx, event['duration'], left=event['start'], color=C_MAP.get(event['stage'], 'gray'))

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Order')
    ax.set_title(f'Order Timeline ({c.N_LINECOOKS} linecooks, {c.N_ASSEMBLERS} assemblers, {c.N_HELPERS} helpers)')
    ax.grid(True, which='major', axis='x', linestyle='--', linewidth=0.5)
    ax.vlines(c.ORDER_TIME, ymin=-1, ymax=200, color='navy', linestyles='dashed', linewidth=2, label='End of orders')

    vline_legend = Line2D(
        [], [],
        color='navy', linestyle='dashed', linewidth=2,
        label='End of orders'
    )

    legend_elements = [Patch(facecolor=color, label=label) for label, color in C_MAP.items()]
    legend_elements.append(vline_legend)
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    # plt.show()
    plt.savefig(path)

def create_resource_chart(timeline_events, path):
    """Creates and saves a chart of resource utilization over time."""
    timeline_events['duration'] = timeline_events['end'] - timeline_events['start']

    resource_lanes = []
    for i in range(c.N_LINECOOKS):
        resource_lanes.append(f'prep_{i + 1}')
    for i in range(c.N_ASSEMBLERS):
        resource_lanes.append(f'assembly_{i + 1}')
    for i in range(c.N_HELPERS):
        resource_lanes.append(f'helper_{i + 1}')

    lane_ends = {lane: 0 for lane in resource_lanes}

    timeline_events['lane'] = -1

    for index, event in timeline_events.sort_values(by='start').iterrows():
        resource_type = event['resource']
        for i, lane in enumerate(resource_lanes):
            if resource_type in lane and event['start'] >= lane_ends[lane]:
                timeline_events.at[index, 'lane'] = i
                lane_ends[lane] = event['end']
                break

    work_events = timeline_events[timeline_events['lane'] != -1]

    fig, ax = plt.subplots(figsize=(20, 10))

    y_pos = range(len(resource_lanes))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(resource_lanes)
    ax.set_ylim(-0.5, len(resource_lanes) - 0.5)

    for index, event in work_events.iterrows():
        ax.barh(event['lane'], event['duration'], left=event['start'], color=C_MAP.get(event['stage'], 'gray'))

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Resource')
    ax.set_title(f'Resource Utilization ({c.N_LINECOOKS} linecooks, {c.N_ASSEMBLERS} assemblers, {c.N_HELPERS} helpers)')
    ax.grid(True, which='major', axis='x', linestyle='--', linewidth=0.5)
    ax.vlines(c.ORDER_TIME, ymin=-1, ymax=100, color='navy', linestyles='dashed', linewidth=2, label='End of orders')

    vline_legend = Line2D(
        [], [],
        color='navy', linestyle='dashed', linewidth=2,
        label='End of orders'
    )

    legend_elements = [Patch(facecolor=color, label=label) for label, color in C_MAP.items() if
                       label in work_events['stage'].unique()]
    legend_elements.append(vline_legend)
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    # plt.show()
    plt.savefig(path)

def create_container_level_chart(all_levels_df, path, cmap):
    """Creates and saves a chart of container fill levels over time for multiple scenarios."""
    containers = sorted(all_levels_df['container'].unique())
    scenarios = sorted(all_levels_df['scenario'].unique())

    n_containers = len(containers)
    n_cols = 2
    n_rows = math.ceil(n_containers / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows), sharex=True)
    axes = axes.flatten()

    for i, container_name in enumerate(containers):
        ax = axes[i]
        max_capacity_for_container = 0

        for scenario_name in scenarios:
            scenario_container_data = all_levels_df[
                (all_levels_df['container'] == container_name) &
                (all_levels_df['scenario'] == scenario_name)
            ]

            if not scenario_container_data.empty:
                ax.step(scenario_container_data['time'], scenario_container_data['level'], where='post', label=scenario_name, c=cmap[scenario_name])

                current_cap = scenario_container_data['container_capacity'].iloc[0]
                if current_cap > max_capacity_for_container:
                    max_capacity_for_container = current_cap

        ax.set_title(f'Fill Level of {container_name}')
        ax.set_ylabel('Fill Level')
        ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5)
        ax.set_ylim(0, max_capacity_for_container * 1.1 if max_capacity_for_container > 0 else c.CONTAINER_SIZE * 1.1)
        ax.set_xlim(0, c.ORDER_TIME + 3600)  # Extend the window by 1 hour

    for i in range(n_containers, len(axes)):
        fig.delaxes(axes[i])

    fig.supxlabel('Time (seconds)')
    fig.suptitle('Container Fill Levels Across Scenarios', fontsize=16)

    handles = [Line2D([], [], c=cmap[s], label=s) for s in scenarios]
    fig.legend(handles=handles, loc='upper right')

    plt.tight_layout()
    plt.savefig(path)

def _plot_comparison_barchart(data, y_col, title, y_label, path):
    """
    Helper function to create a single comparison bar plot.
    It visualizes the percentage change of a given metric against a baseline for different scenarios.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = ['#d9534f' if x > 0 else '#5cb85c' for x in data[y_col]]
    bars = ax.bar(data['name'], data[y_col], color=colors)

    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=16, pad=20)
    ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5)
    plt.xticks(rotation=45, ha="right")

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.1f}%', va='bottom' if yval > 0 else 'top',
                ha='center')

    ax.axhline(0, color='black', linewidth=0.8)

    plt.tight_layout()
    plt.savefig(path)

def create_scenario_comparison_charts(df):
    """
    Creates and saves bar plots comparing simulation scenarios against the baseline.
    Two plots are generated:
    1. Percentage change in average total leftovers vs. baseline.
    2. Percentage change in average assembly wait time vs. baseline.
    """
    baseline = df[df['scenario_type'] == 'baseline'].iloc[0]
    scenarios = df[df['scenario_type'] != 'baseline'].copy()

    baseline_leftovers = baseline['avg_total_leftovers']
    if baseline_leftovers == 0:
        baseline_leftovers = 1
    scenarios['leftovers_pct_change'] = ((scenarios['avg_total_leftovers'] - baseline_leftovers) / baseline_leftovers) * 100

    baseline_wait_time = baseline['avg_assembly_wait_time_for_refill']
    if baseline_wait_time == 0:
        baseline_wait_time = 1
    scenarios['wait_time_pct_change'] = ((scenarios['avg_assembly_wait_time_for_refill'] - baseline_wait_time) / baseline_wait_time) * 100

    _plot_comparison_barchart(
        data=scenarios,
        y_col='leftovers_pct_change',
        title='Percentage Change in Average Leftover Ingredients vs. Baseline',
        y_label='Change in Leftovers (%)',
        path='figures/leftovers_comparison.png'
    )

    _plot_comparison_barchart(
        data=scenarios,
        y_col='wait_time_pct_change',
        title='Percentage Change in Average Assembly Wait Time vs. Baseline',
        y_label='Change in Wait Time (%)',
        path='figures/wait_time_comparison.png'
    )

if __name__ == '__main__':
    burgenerator_instance = burgenerator_storage.run_single_simulation(seed=c.SEED)
    timeline_df = pd.DataFrame(burgenerator_instance.timeline_events)

    create_gantt_chart(timeline_df.copy(), f'figures/gantt_chart_{c.N_LINECOOKS}_{c.N_ASSEMBLERS}_{c.N_HELPERS}.png')
    create_resource_chart(timeline_df.copy(),
                          f'figures/resource_chart_{c.N_LINECOOKS}_{c.N_ASSEMBLERS}_{c.N_HELPERS}.png')

    all_scenario_results = burgenerator_storage.run_and_collect_scenario_results(return_levels=True)
    all_levels_dfs = []
    scenario_summary_data = []
    for scenario_run in all_scenario_results:
        scenario_name = scenario_run['name']
        scenario_type = scenario_run['type']
        results = scenario_run['results']
        levels = results.get('container_levels')

        scenario_summary_data.append({
            'name': scenario_name,
            'scenario_type': scenario_type,
            'avg_total_leftovers': results['avg_total_leftovers'],
            'avg_assembly_wait_time_for_refill': results['avg_assembly_wait_time_for_refill']
        })

        if levels:
            levels_df = pd.DataFrame(levels)
            levels_df['scenario'] = scenario_name
            levels_df['scenario_type'] = scenario_type
            levels_df['container_capacity'] = results.get('container_size', c.CONTAINER_SIZE)
            all_levels_dfs.append(levels_df)

    if all_levels_dfs:
        full_levels_df = pd.concat(all_levels_dfs, ignore_index=True)
        threshold_df = full_levels_df[full_levels_df['scenario_type'].isin(['baseline', 'threshold'])]
        create_container_level_chart(threshold_df, 'figures/container_chart_threshold_scenarios.png', C_MAP_THRESHOLD)
        capacity_df = full_levels_df[full_levels_df['scenario_type'].isin(['baseline', 'capacity'])]
        create_container_level_chart(capacity_df, 'figures/container_chart_capacity_scenarios.png', C_MAP_CAPACITY)

    scenario_summary_df = pd.DataFrame(scenario_summary_data)
    create_scenario_comparison_charts(scenario_summary_df)