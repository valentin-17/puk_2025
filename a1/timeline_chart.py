"""
To save time this file was mostly written by Google's Gemini CLI. The quality of the code produced by the agent is
impressive in my opionion. Only minor adjustments in naming and organizing the code have to be made to receive usable
plotting utility functions.

The file hosts two methods. The create_gantt_chart method holds all burger orders on the x-axis and displays the entire
simulation run as a timeline. Here we can observe how long an order waits until it gets processed. It is important to
remember that this plot shows the process from the burger order perspective. prep_wait time does not suggest that the
prep station has to wait all the time.

To view the process at Theke 3 from the resource perspective (i.e. the workers) the method create_resource_chart is more
useful. The plot shows the utilization of the linecooks as well as the utilization of the burgermaster. Here it becomes
clear that the burgermaster has high downtime and often waits for the preparation of the warm ingredients. This also
visually confirms the numerical bottleneck that has been identified in the analysis step of the burgenerator.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import a1.config as c
from matplotlib.patches import Patch

from a1 import burgenerator, config


C_MAP = {
        'prep_wait': '#afe1f5',
        'prep_work': '#69a0cb',
        'assembly_wait_to_begin': '#ffd700',
        'assembly_wait': '#d1a7ef',
        'assembly_work': '#ba55aa',
        'assembly_packing': '#840853'
    }

def create_gantt_chart(timeline_events, path):
    """Creates and saves a Gantt chart of the burger simulation."""
    timeline_events['duration'] = timeline_events['end'] - timeline_events['start']

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
    ax.set_title(f'Order Timeline ({c.N_LINECOOKS} linecooks, {c.N_ASSEMBLERS} assemblers)')
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
    #plt.show()
    plt.savefig(path)

def create_resource_chart(timeline_events, path):
    """Creates and saves a chart of resource utilization over time."""
    timeline_events['duration'] = timeline_events['end'] - timeline_events['start']
    
    resource_lanes = []
    for i in range(config.N_LINECOOKS):
        resource_lanes.append(f'prep_station_{i+1}')
    for i in range(config.N_ASSEMBLERS):
        resource_lanes.append(f'assembly_{i+1}')

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
    ax.set_title(f'Resource Utilization ({c.N_LINECOOKS} linecooks, {c.N_ASSEMBLERS} assemblers)')
    ax.grid(True, which='major', axis='x', linestyle='--', linewidth=0.5)
    ax.vlines(c.ORDER_TIME, ymin=-1, ymax=100, color='navy', linestyles='dashed', linewidth=2, label='End of orders')

    vline_legend = Line2D(
        [], [],
        color='navy', linestyle='dashed', linewidth=2,
        label='End of orders'
    )

    legend_elements = [Patch(facecolor=color, label=label) for label, color in C_MAP.items() if label in work_events['stage'].unique()]
    legend_elements.append(vline_legend)
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    #plt.show()
    plt.savefig(path)

if __name__ == '__main__':
    burgenerator_instance = burgenerator.run_single_simulation(seed=config.SEED)
    timeline_df = pd.DataFrame(burgenerator_instance.timeline_events)

    create_gantt_chart(timeline_df.copy(), 'figures/gantt_chart_2_1.png')
    create_resource_chart(timeline_df.copy(), 'figures/resource_chart_2_1.png')
