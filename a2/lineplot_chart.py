
import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def create_line_plot(data_path, output_path):
    """Creates a line plot of container levels over time."""
    df = pd.read_csv(data_path)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(20, 10))

    sns.lineplot(data=df, x='time', y='level', hue='container', ax=ax, errorbar=None)

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Stock Level')
    ax.set_title('Container Stock Levels Over Time')
    ax.legend(title='Ingredient')
    ax.grid(True, which='major', axis='x', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = 'container_levels.csv'
    
    output_filename = 'a2/figures/container_levels.png'
    create_line_plot(csv_path, output_filename)
    print(f"Line plot saved to {output_filename}")
