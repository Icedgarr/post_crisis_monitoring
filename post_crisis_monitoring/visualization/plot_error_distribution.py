import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def plot_error_distribution(data, column, xlabel, xticks, xticks_labels=None, save_path=None, output_format='tiff'):
    plt.figure(figsize=(6, 4))
    plt.hist(data[column], weights=np.ones(len(data)) / len(data),
             edgecolor='black', bins=50)
    plt.axvline(x=data[column].mean(), color='r', linestyle=':', label='mean')
    plt.axvline(x=data[column].median(), color='g', linestyle='-', label='median')
    plt.axvline(x=data[column].mean() + data[column].std(), color='r', linestyle='--',
                label='mean+/-std')
    plt.axvline(x=data[column].mean() - data[column].std(), color='r', linestyle='--')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel('Percentage (%)', fontsize=14)
    plt.xticks(ticks=xticks, labels=xticks_labels, fontsize=12)
    plt.legend(fontsize=12)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', format=output_format, dpi=400)
    plt.show()
