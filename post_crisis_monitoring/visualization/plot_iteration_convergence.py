import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def plot_iteration_convergence(data, column, xticks, xticks_labels=None, save_path=None, output_format='tiff'):
    median_itera = int(data[column].median())
    quant_75_itera = int(data[column].quantile(0.75))
    plt.figure(figsize=(10, 6))
    plt.hist(data[column],
             weights=np.ones(len(data)) / len(data),
             edgecolor='black', bins=50, label='')
    plt.axvline(x=quant_75_itera, color='r', linestyle=':', label=f'75% quantile ({quant_75_itera})')
    plt.axvline(x=median_itera, color='g', linestyle='-', label=f'median ({median_itera})')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.xlabel('Number of iterations', fontsize=14)
    plt.ylabel('Percentage (%)', fontsize=14)
    plt.xticks(ticks=xticks, labels=xticks_labels, fontsize=12)
    plt.legend(fontsize=12)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', format=output_format, dpi=400)
    plt.show()
