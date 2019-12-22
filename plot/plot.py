import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


path_to_results = '../results/'


def setup_printer():
    sns.set()

    pd.set_option('display.max_colwidth', -1)
    pd.set_option('display.max_columns', 30)
    pd.set_option('expand_frame_repr', False)


def load_data(file):
    return pd.read_json(file).T


def calculate_speedup(df, base='CPU CSR'):
    speedup = df.copy()
    columns = list(df)

    for col in columns:
        speedup[col] = df[base] / df[col]

    return speedup


setup_printer()

source_df = load_data('{}/result.json'.format(path_to_results))
float_speedup = calculate_speedup(source_df).reset_index()


def join_show(df, target, filename=''):
    sns.jointplot(data=df, x='block size', y=target, kind='reg', height=14, joint_kws={'scatter_kws': dict(alpha=0.6)})

    if filename:
        plt.legend(prop={'size': 12})
        plt.savefig(filename, dpi=200, bbox_inches='tight')
    else:
        plt.legend(prop={'size': 22})
        plt.show()


fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(ax=ax, data=float_speedup, x='index', y='GPU CSR', legend='full', label='GPU CSR')
sns.lineplot(ax=ax, data=float_speedup, x='index', y='GPU BCSR (row major, thread per row)', legend='full', label='GPU BCSR (row major, thread per row)')
ax.set(xlabel='Block size', ylabel='Speedup')
plt.legend(prop={'size': 12})
plt.show()

