import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np

from a4.burgernerator_job_shop import run_simulations


def _aggregate_machine_table(stats_list):
    frames = []
    for s in stats_list:
        if 'machine_variant_table' in s:
            frames.append(s['machine_variant_table'])

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames)
    grouped = (
        df.groupby('machine')
        .agg({
            'util_percent': 'mean',
            'avg_wait_s': 'mean',
            'avg_work_s': 'mean',
            'total_busy_s': 'mean'
        })
        .reset_index()
    )

    return grouped


def plot_machine_utilization(stats_list, path):
    df = _aggregate_machine_table(stats_list)
    df_sorted = df.sort_values('util_percent', ascending=False)

    plt.figure(figsize=(9, 5))
    plt.bar(df_sorted['machine'], df_sorted['util_percent'],
            color='#69a0cb')

    plt.ylabel('Utilization (%)')
    plt.title('Average Machine Utilization')
    plt.tight_layout()
    #plt.show()
    plt.savefig(path)


def plot_machine_wait_times(stats_list, path):
    df = _aggregate_machine_table(stats_list)
    df_sorted = df.sort_values('avg_wait_s', ascending=False)

    plt.figure(figsize=(9, 5))
    plt.bar(df_sorted['machine'], df_sorted['avg_wait_s'],
            color='#69a0cb')

    plt.ylabel('Average Wait Time (s)')
    plt.title('Average Waiting Time per Machine')
    plt.tight_layout()
    #plt.show()
    plt.savefig(path)


def plot_sojourn_hist(stats_list, path):
    sojourns = []
    for s in stats_list:
        raw = s['order_stats'].get('sojourn_time', [])
        sojourns.extend([t / 60.0 for t in raw])

    plt.figure(figsize=(9, 5))
    plt.hist(sojourns, bins=30, color='#afe1f5', edgecolor='#69a0cb')

    plt.xticks()
    plt.xlabel('Sojourn Time (min)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Job Sojourn Times')
    plt.tight_layout()
    #plt.show()
    plt.savefig(path)


def plot_sojourn_by_burger_and_rule(stats_by_rule, path):
    """
    Plot average sojourn time (minutes) per burger type, grouped by scheduling rule.
    Accepts either a dict of {rule: [stats]} or a plain list of stats from one rule.
    """
    rows = []

    def add_rows(rule_label, stats_list):
        for stat in stats_list:
            types = stat.get('order_stats', {}).get('burger_type', [])
            sojourns = stat.get('order_stats', {}).get('sojourn_time', [])
            for bt, sj in zip(types, sojourns):
                rows.append({
                    'burger_type': bt,
                    'rule': rule_label,
                    'sojourn_min': sj / 60.0
                })

    if isinstance(stats_by_rule, dict):
        for rule_label, stats_list in stats_by_rule.items():
            add_rows(rule_label, stats_list)
    else:
        label = stats_by_rule[0].get('queue_rule', 'selected') if stats_by_rule else 'selected'
        add_rows(label, stats_by_rule)

    df = pd.DataFrame(rows)
    if df.empty:
        return

    grouped = (
        df.groupby(['burger_type', 'rule'])['sojourn_min']
        .mean()
        .reset_index()
    )

    burger_types = sorted(grouped['burger_type'].unique())
    rules = sorted(grouped['rule'].unique())

    x = np.arange(len(burger_types))
    width = 0.8 / max(len(rules), 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    palette = ['#69a0cb', '#7ccba2', '#f4a261', '#e76f51', '#a07be3', '#4d908e']

    for idx, rule in enumerate(rules):
        vals = (
            grouped[grouped['rule'] == rule]
            .set_index('burger_type')
            .reindex(burger_types)['sojourn_min']
        )
        ax.bar(x + (idx - (len(rules) - 1) / 2) * width,
               vals,
               width=width,
               label=rule,
               color=palette[idx % len(palette)])

    ax.set_xticks(x)
    ax.set_xticklabels(burger_types)
    ax.set_ylabel('Average Sojourn Time (min)')
    ax.set_xlabel('Burger Type')
    ax.set_title('Sojourn Time by Burger Type and Scheduling Rule')
    ax.legend(title='Scheduling')
    plt.tight_layout()
    plt.savefig(path)


def plot_queue_length_timeline(timeline_events, path):
    machines = sorted({ev.get('resource') for ev in timeline_events
                           if 'wait_end' in ev.get('stage', '')})

    waits_by_machine = {
        m: [ev for ev in timeline_events
            if ev.get('resource') == m and 'wait_end' in ev.get('stage', '')]
        for m in machines
    }

    t_max = max(
        (ev['end'] for ev in timeline_events if 'wait_end' in ev.get('stage', '')),
        default=0,
    )

    times = np.linspace(0, t_max, 300)

    fig, axes = plt.subplots(len(machines), 1, figsize=(9, 5 * len(machines)))

    if len(machines) == 1:
        axes = [axes]

    for ax, m in zip(axes, machines):
        waits = waits_by_machine[m]
        qlens = []

        for t in times:
            active_waits = sum(1 for ev in waits if ev['start'] <= t <= ev['end'])
            qlens.append(active_waits)

        ax.step(times, qlens, label=m, color='#69a0cb')
        ax.set_ylabel("Queue Length")
        ax.set_title(f"Machine {m}")

        ax.xaxis.set_major_locator(mticker.MultipleLocator(1800))
        ax.xaxis.set_major_formatter(lambda x, _: f"{int(x // 3600):02d}:{int((x % 3600) // 60):02d}" if (x % 1800) == 0 else "")

        ax.xaxis.set_minor_locator(mticker.MultipleLocator(900))

        ax.yaxis.set_major_locator(mticker.MultipleLocator(1))

        ax.set_ylim(0, 7)

    axes[-1].set_xlabel("Time (hour)")
    plt.tight_layout()
    #plt.show()
    plt.savefig(path)


def plot_gantt(timeline_events, path):
    df = pd.DataFrame([
        e for e in timeline_events
        if '_work' in e.get('stage', '')
    ])

    df['duration'] = df['end'] - df['start']

    groups = sorted(df['resource'].unique())

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, m in enumerate(groups):
        segs = df[df['resource'] == m]
        for _, row in segs.iterrows():
            ax.broken_barh([(row['start'], row['duration'])],
                           (i - 0.4, 0.8), color='#69a0cb')

    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups)
    ax.set_xlabel('Time (s)')
    ax.set_title('Machine Activity Timeline')

    ax.xaxis.set_major_locator(mticker.MultipleLocator(1800))
    ax.xaxis.set_major_formatter(
        lambda x, _: f"{int(x // 3600):02d}:{int((x % 3600) // 60):02d}" if (x % 1800) == 0 else "")

    ax.xaxis.set_minor_locator(mticker.MultipleLocator(900))

    plt.tight_layout()
    #plt.show()
    plt.savefig(path)


if __name__ == '__main__':
    stats_by_rule = run_simulations(return_by_rule=True)

    # flatten stats across rules for the generic plots
    flat_stats = []
    if isinstance(stats_by_rule, dict):
        for stats in stats_by_rule.values():
            flat_stats.extend(stats)
    else:
        flat_stats = stats_by_rule

    if flat_stats:
        plot_machine_utilization(flat_stats, 'figures/machine_utilization.png')
        plot_machine_wait_times(flat_stats, 'figures/machine_wait_times.png')
        plot_sojourn_hist(flat_stats, 'figures/sojourn_times.png')
        plot_queue_length_timeline(flat_stats[0]['timeline_events'], 'figures/queue_lengths.png')
        plot_gantt(flat_stats[0]['timeline_events'], 'figures/machine_utilization_gantt.png')

    plot_sojourn_by_burger_and_rule(stats_by_rule, 'figures/sojourn_by_burger_and_rule.png')
