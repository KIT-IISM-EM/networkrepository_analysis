"""

.. moduleauthor:: Fabian Ball <fabian.ball@kit.edu>
"""
from __future__ import division, print_function, unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import map, range

import argparse
from decimal import Decimal as D, getcontext
import math
import os
from string import Template
import io
import sys

import matplotlib
# http://matplotlib.org/users/pgf.html
pgf_with_pdflatex = {
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": [
         "\\usepackage[utf8x]{inputenc}",
         "\\usepackage[T1]{fontenc}",
         # "\\usepackage{mathptmx}",
         ]
}
matplotlib.rcParams.update(pgf_with_pdflatex)
matplotlib.use("pgf")

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from pandas.tools.plotting import scatter_matrix

pd.set_option('display.width', 200)
pd.set_option('display.float_format', '{:,.5g}'.format)

plt.style.use('ggplot')  # ggplot-like style

# http://matplotlib.org/users/customizing.html
params = {
    "font.family": "serif",
    "font.serif": ["Palatino"],  # use latex default serif font
    # "font.sans-serif": ["DejaVu Sans"],  # use a specific sans-serif font
    "font.size": 10.0,  # Set the global font-size
    "axes.labelsize": "small",  # Set the axes font-size
    "legend.fontsize": "small",  # Set the legend font-size
    "xtick.labelsize": "small",
    "ytick.labelsize": "small",
    # "figure.figsize": [6.25, 4.75],  # [inch^2] = 158,75 x 120.65 [mm^2]
    # Set the total figure size:
    # These dimensions seem to fit quite well, two figures fit on one page with some caption text
    # \textwidth is 160mm
    "figure.figsize": [6.0, 3.5],  # [inch^2] = 158,75 x 101,6 [mm^2]
    "figure.subplot.bottom": 0.11,

}
matplotlib.rcParams.update(params)

getcontext().prec = 5  # Set precision for Decimal
getcontext().capitals = 0  # No capital "E" ("e" instead)


DROP_DUPLICATES = True


def PathType(path_str):
    """
    Determine if the given string is an existing path in the file system.

    :param path_str:
    :return:
    """
    orig = path_str
    path_str = os.path.expanduser(path_str)  # Expand user path if necessary
    path_str = os.path.abspath(path_str)

    if os.path.exists(path_str):
        return path_str
    else:
        raise argparse.ArgumentError(None, '"{}" is not a valid path'.format(orig))


def latex_table(content, caption, short_caption, label, label_prefix='tab:'):
    """
    Return a LaTeX table-environment.

    :param content: A valid LaTeX tabular-environment
    :param caption: A caption string
    :param short_caption: A short caption string
    :param label: A valid label string
    :param label_prefix: A label prefix
    :return: LaTeX table-environment string
    """
    LATEX_TABLE_TPL = """
\\begin{table}[htb]
\centering
{
\small
$content
}
\caption[$short_caption]{$caption}\label{$label}
\end{table}
    """
    tpl = Template(LATEX_TABLE_TPL)
    return tpl.substitute(content=content.replace('%', '\%'),
                          caption=caption,
                          short_caption=short_caption,
                          label='{}{}'.format(label_prefix, label))


def transform_stats_df(df):
    if DROP_DUPLICATES:
        df.drop_duplicates(['Graph Name'], inplace=True)

    df = df[['Graph Name', 'Type']]  # Create a new DF with only the relevant columns

    return df


def transform_analysis_df(df):
    df['filename'] = df['name']  # Duplicate row: 'name' contains <graphname>.<extension>
    df['file_extension'] = df['name'].apply(lambda s: s.split('.')[-1])  # Create row with file extension only
    df['name'] = df['name'].apply(lambda s: '.'.join(s.split('.')[:-1]))  # Remove file extension from name
    df['name'] = df['name'].apply(lambda s: s.replace('_', '-'))  # Underscore in filename -> minus in graph name

    if DROP_DUPLICATES:
        df.drop_duplicates(['name'], inplace=True)

    # Compute the automorphism group size using Decimal
    df.rename(columns={'aut_group_size': 'aut_group_size_mult'}, inplace=True)
    df['aut_group_size'] = [D(row['aut_group_size_mult']) * D(10) ** D(row['aut_group_size_exp'])
                            for _, row in df.iterrows()]
    # Set the numpy nan instead of decimal NaN
    df['aut_group_size'] = df['aut_group_size'].apply(lambda v: np.nan if v.is_nan() else v)

    return df


def transform_merged_df(df):
    df.drop(['Graph Name'], axis=1, inplace=True)  # Remove the duplicate name column
    # Reorder columns
    df = df[['name', 'Type', 'n', 'm', 'density', 'modularity', 'aut_group_size',
             # 'aut_group_size_mult', 'aut_group_size_exp',
             'num_generators', 'num_orbits']]
    df = df.sort_values(['name'])  # Sort rows by name
    df.index = list(range(len(df)))  # Set a new index after the sorting

    def rename_rt_graphs(entry):
        if entry == 'retweet_graphs':
            return 'rt'
        else:
            return entry

    df['Type'] = df['Type'].apply(rename_rt_graphs)

    df['redundancy_m'] = (df['num_orbits'] - 1) / df['n']  # Add the 'network redundancy'
    df['redundancy'] = 1 - (df['num_orbits'] - 1) / (df['n'] - 1)  # Add the normalized 'network redundancy'

    return df


def print_graph_stats_statistics(df_s):
    type_names = {'bhoslib': 'BHOSLIB',  # Benchmarks with Hidden Optimum Solutions for Graph Problems
                  'bio': 'Biological Networks',
                  'bn': 'Brain Networks',
                  'ca': 'Collaboration Networks',
                  'chem': 'Cheminformatics',
                  'dimacs': 'DIMACS',  # Center for Discrete Mathematics and Theoretical Computer Science
                  'dimacs10': 'DIMACS10',
                  'dynamic': 'Dynamic Networks',
                  'eco': 'Ecology Networks',
                  'ia': 'Interaction Networks',
                  'inf': 'Infrastructure Networks',
                  'massive': 'Massive Network Data',
                  'misc': 'Miscellaneous Networks',
                  'rec': 'Recommendation Networks',
                  'retweet_graphs': 'Retweet Networks',
                  'rt': 'Retweet Networks',
                  'sc': 'Scientific Computing',
                  'soc': 'Social Networks',
                  'socfb': 'Facebook Networks',
                  'tech': 'Technological Networks',
                  'tscc': 'Temporal Reachability Networks',
                  'web': 'Web Graphs',
                  }

    # Flatten inconsistency: retweet networks are sometimes typed 'rt', sometimes 'retweet_graphs'
    def rename_rt_graphs(entry):
        if entry == 'retweet_graphs':
            return 'rt'
        else:
            return entry

    df_s['Type'] = df_s['Type'].apply(rename_rt_graphs)

    gb_type = df_s.groupby(['Type'])
    records = {'sum': {'Short Type Name': '', 'Count': len(df_s), 'Type Name': '$\sum$'}}

    for name, group_df in gb_type:
        records[name] = {'Short Type Name': name, 'Count': len(group_df), 'Type Name': type_names.get(name, 'NA')}

    df_nodup = df_s.drop_duplicates(['Graph Name'])
    records['sum']['Count (w/o duplicates)'] = len(df_nodup)

    gb_type = df_nodup.groupby(['Type'])

    for name, group_df in gb_type:
        records[name]['Count (w/o duplicates)'] = len(group_df)

    df = pd.DataFrame.from_records(records.values())
    df = df[['Type Name', 'Short Type Name', 'Count', 'Count (w/o duplicates)']]
    # Set the 'double' name for retweet graphs
    df.loc[df[df['Short Type Name'] == 'rt'].index, 'Short Type Name'] = 'rt/retweet\_networks'
    df = df.sort_values(['Type Name'])
    df.index = list(range(len(df)))  # Set a new index after the sorting
    print(latex_table(df.to_latex(escape=False, index=False),
                      'Number of data sets for different network types on \\texttt{networkrepository.com}. '
                      'Duplicates were dropped by graph name',
                      'Data set overview for \\texttt{networkrepository.com}',
                      'networkrepository.com_statistics'))


def _decimal_statistics(series):
    # Custom implementation to compute the values for pd.DataFrame.describe() for Decimals
    cleaned_series = series.dropna()
    decimal_list = sorted([D(v) for v in cleaned_series])

    count = len(decimal_list)

    if count == 0:
        return 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    sum_ = sum(decimal_list)
    mean = sum_ / count
    std = (sum((d - mean)**2 for d in decimal_list) / (count - 1)).sqrt() if count > 1 else np.nan
    min_ = min(decimal_list)
    max_ = max(decimal_list)
    assert min_ == decimal_list[0]
    assert max_ == decimal_list[-1]

    def quantile(p, l):
        n = len(l)
        if (n * p) % 2 == 0:
            idx = int(n * p) - 1  # 0-indexed!
            return (l[idx] + l[idx+1]) / 2
        else:
            idx = int(math.floor(n * p + 1)) - 1  # 0-indexed!
            return l[idx]

    p25 = quantile(0.25, decimal_list)
    p50 = quantile(0.5, decimal_list)
    p75 = quantile(0.75, decimal_list)

    return count, mean, std, min_, p25, p50, p75, max_


def print_total_analysis_statistics(df):
    # Remove all DIMACS10 graphs as they are sure duplicates!
    # df = df[df['Type'] != 'dimacs10']

    duplicates = df[df.duplicated(['name'])]['name']
    # df_total = df.drop_duplicates(['name'])
    df_total = df
    # df_total = df[['name', 'Type', 'n', 'm', 'density', 'modularity', 'aut_group_size',
    #                'num_generators', 'num_orbits']]

    df_stat = df_total.describe()  # Compute standard statistics (count, mean, std, quartiles, min, max)

    aut_grp_size_stat = _decimal_statistics(df_total['aut_group_size'])
    df_stat['aut_group_size'] = aut_grp_size_stat

    # Drop the two columns, as the statistics are not meaningful for those
    df_stat.drop(['num_orbits', 'num_generators'], axis=1, inplace=True)
    # Rename the columns for LaTeX export
    df_stat.rename(columns={'n': '$n$', 'm': '$m$', 'density': r'$\rho$', 'modularity': '$Q$',
                            'aut_group_size': '$|Aut(G)|$', 'redundancy': '$r_G\'$'}, inplace=True)
    # df_stat.apply(lambda v: round(v, 10))
    print('%% Statistics for the complete data (size={})'.format(len(df_total)))
    print('%% (Duplicates: {})'.format(list(duplicates)))
    counts = {'analyzed': len(df_total) - len(df_total[df_total['aut_group_size'].isnull()]),
              'asymmetric': len(df_total[df_total['n'] == df_total['num_orbits']])}
    short_caption = "Analysis statistics for \\texttt{networkrepository.com} data sets"
    caption = '{short_caption}: ${asymmetric}$ of the ${analyzed}$ graphs that ' \
              'were analyzed for symmetry are asymmetric'.format(short_caption=short_caption, **counts)
    print(latex_table(df_stat.to_latex(na_rep='nan', escape=False), caption, short_caption, 'networkrepos_total'))
    # print latex_table(df_stat.to_latex(escape=False), caption, short_caption, 'networkrepos_total')
    print('')


def print_group_analysis_statistics(group_name, df_group):
    duplicates = df_group[df_group.duplicated(['name'])]['name']
    group_df = df_group.drop_duplicates(['name'])

    df_stat = group_df.describe()
    df_stat.drop(['num_orbits', 'num_generators'], axis=1, inplace=True)

    aut_grp_size_stat = _decimal_statistics(df_group['aut_group_size'])
    df_stat['aut_group_size'] = aut_grp_size_stat

    df_stat.rename(columns={'n': '$n$', 'm': '$m$', 'density': r'$\rho$', 'modularity': '$Q$',
                            'aut_group_size': '$|Aut(G)|$', 'redundancy': '$r_G\'$'}, inplace=True)
    print('%% Statistics for type "{}" (group-size={})'.format(group_name, len(group_df)))
    print('%% (Duplicates: {})'.format(list(duplicates)))
    counts = {'analyzed': len(group_df) - len(group_df[group_df['aut_group_size'].isnull()]),
              'asymmetric': len(group_df[group_df['n'] == group_df['num_orbits']])}
    short_caption = "Analysis statistics for category ``%s'' on \\texttt{networkrepository.com}" % group_name
    caption = '{short_caption}: ${asymmetric}$ of the ${analyzed}$ graphs that ' \
              'were analyzed for symmetry are asymmetric'.format(short_caption=short_caption, **counts)
    print(latex_table(df_stat.to_latex(na_rep='nan', escape=False), caption, short_caption,
                      'networkrepos_{}'.format(group_name)))
    print('')


def create_scatterplot(df, attr1, attr2, xlabel=None, ylabel=None, xlim=None, ylim=None, c=None, cmap=None):
    if not xlabel:
        xlabel = attr1

    if not xlabel:
        ylabel = attr2

    fig = plt.figure()
    ax = plt.subplot('111')
    # ax.scatter(df['density'], df['modularity'], c=df['aut_group_size'].apply(lambda x: x == 1),
    #            s=20, marker='.', linewidth=0)
    # ax.scatter(df['density'], df['modularity'], c=df['Type'].apply(hash), s=20, marker='.', linewidth=0)
    paths = ax.scatter(df[attr1], df[attr2], c=c, s=20, marker='.', linewidth=0, cmap=cmap)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    return fig, ax


def create_scatterplots(df, args):
    # # Complete scatter matrix
    # # axes = scatter_matrix(scatter_df[['n', 'm', 'density', 'modularity', 'redundancy']])
    # axes = scatter_matrix(df[['density', 'modularity', 'redundancy']], c=df['Type'].apply(hash))
    # # for x, y in ((x, y) for x in xrange(0, 5) for y in xrange(2, 5)):
    # #     if x == y:
    # #         continue
    # #     axes[x, y].set_xlim((-0.1, 1.1))
    # # for x, y in ((x, y) for x in xrange(2, 5) for y in xrange(0, 5)):
    # #     if x == y:
    # #         continue
    # #     axes[x, y].set_ylim((-0.1, 1.1))
    # plt.savefig(os.path.join(args.target_dir, 'networkrepository_scatter_complete.pgf'))
    # plt.close()

    # *** Modularity and redundancy ***
    lim = (-.02, 1.02)
    fig, ax = create_scatterplot(df, 'modularity', 'redundancy', c=df['Type'].apply(hash),
                                 xlabel='$Q$', ylabel='$r_G\'$', xlim=lim, ylim=lim)

    ax.legend()
    fig.tight_layout()

    fig.savefig(os.path.join(args.target_dir, 'networkrepository_scatter_Q_rG.pgf'))
    plt.close()

    # # |Aut(G)| and redundancy
    # lim = (-.02, 1.02)
    # df['aut_g_size_exp'] = df['aut_group_size_exp'] + df['aut_group_size_mult'].apply(math.log10).apply(math.floor)
    # fig, ax = create_scatterplot(df, 'aut_g_size_exp', 'redundancy', c=df['Type'].apply(hash),
    #                              xlabel='$b$', ylabel='$r_G\'$',
    #                              xlim=(0, df['aut_g_size_exp'].max()), ylim=lim)
    #
    # ax.set_xscale('symlog')
    # ax.legend()
    # fig.tight_layout()
    #
    # fig.savefig(os.path.join(args.target_dir, 'networkrepository_scatter_AutG_rG.pgf'))
    # plt.close()

    # *** Modularity and density ***
    lim = (-.02, 1.02)
    fig, ax = create_scatterplot(df, 'modularity', 'density',  # c=df['Type'].apply(hash),
                                 xlabel='$Q$', ylabel='$\\rho$', xlim=lim, ylim=lim)

    # ax.legend()
    fig.tight_layout()

    fig.savefig(os.path.join(args.target_dir, 'networkrepository_scatter_Q_rho.pgf'))
    plt.close()

    # *** Redundancy and Type ***
    gb_type = df.groupby(['Type'])
    type_dict = {name: idx for idx, (name, _) in enumerate(sorted(gb_type, key=lambda e: e[0]))}
    size_dict = {name: len(group) for name, group in gb_type}
    sym_size_dict = {name: len(group[group['n'] > group['num_orbits']]) for name, group in gb_type}

    fig = plt.figure()
    ax = plt.subplot('111')
    paths = ax.scatter(df['redundancy'], df['Type'].apply(lambda t: -type_dict[t]), s=20, marker='.', linewidth=0)

    ax.set_xlabel("$r_G'$")
    ax.set_ylabel("Type")

    ax.set_yticks(sorted([-x for x in type_dict.values()]))
    labels = [u"{} ({}/{})".format(name, size_dict[name], sym_size_dict[name])
              for name in sorted(type_dict.keys(), key=lambda l: -type_dict[l])]

    ax.set_yticklabels(labels)

    ax.set_xlim(-.02, 1.02)
    ax.set_ylim(top=.5,  # Add some space add the top
                bottom=-(len(type_dict) - .5)  # number of entries from 0 to -len(type_dict) + 1 => remove some space
                )

    fig.tight_layout()

    fig.savefig(os.path.join(args.target_dir, 'networkrepository_scatter_rG_type.pgf'))
    plt.close()


def create_histogram_logx(df, attr, bins, latex=True):
    attr_output = attr if not latex else "${}$".format(attr)

    xlim = (0, bins[-1] + bins[-2])

    fig = plt.figure()
    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2)
    ax1.set_xscale('symlog')  # Allows negative values => we only have positive ones but want a real lower bound of 0

    # Create a histogram plot for the complete data
    plt.hist(df[~df['num_orbits'].isnull()][attr], bins=bins, log=True, label="{} for all graphs".format(attr_output))

    # Add the histogram plot for symmetric graphs
    plt.hist(df[(df['n'] > df['num_orbits']) & ~df['num_orbits'].isnull()][attr], bins=bins, log=True,
             label="{} for symmetric graphs".format(attr_output))

    ax1.set_xlabel(attr_output)
    ax1.set_ylabel("Frequency")
    ax1.set_xlim(*xlim)  # Set limits, on the right, add some small space
    ax1.legend(loc=0)
    ax1.set_yscale('linear')  # "Reset" y axis to linear scale

    ax2 = plt.subplot2grid((4, 1), (2, 0))
    ax2.set_xscale('symlog')
    plt.boxplot(df[~df['num_orbits'].isnull()][attr], vert=False, widths=.8, sym='b|')
    ax2.yaxis.set_ticklabels([])
    ax2.set_xlim(*xlim)
    ax2.set_ylim(.4, 1.6)
    ax2.set_xlabel("Box-plot for all graphs")

    ax3 = plt.subplot2grid((4, 1), (3, 0))
    ax3.set_xscale('symlog')
    df2 = df[(df['n'] > df['num_orbits']) & ~df['num_orbits'].isnull()]
    df2.index = list(range(len(df2)))
    plt.boxplot(df2[attr], vert=False, widths=.8, sym='b|')
    ax3.yaxis.set_ticklabels([])
    ax3.set_xlim(*xlim)
    ax3.set_ylim(.4, 1.6)
    ax3.set_xlabel("Box-plot for symmetric graphs")

    fig.tight_layout()

    return fig, (ax1, ax2, ax3)


def create_histogram_and_boxplot(df, attr, num_bins=20, rng=(0, 1), margin=0.02, attr_name=None, xlabel=None):
    if not attr_name:
        attr_name = attr

    if not xlabel:
        xlabel = attr_name

    bins = [i / num_bins * (rng[1] - rng[0]) for i in range(num_bins + 1)]

    width = rng[1] - rng[0]
    xlim = (rng[0] - margin*width, rng[1] + margin*width)

    fig = plt.figure()
    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2)

    # Create a histogram plot for the complete data
    plt.hist(df[~df['num_orbits'].isnull()][attr], bins=bins, label="{} for all graphs".format(attr_name))

    # Add the histogram plot for symmetric graphs
    plt.hist(df[(df['n'] > df['num_orbits']) & ~df['num_orbits'].isnull()][attr], bins=bins,
             label="{} for symmetric graphs".format(attr_name))

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Frequency")
    ax1.set_xlim(*xlim)  # Add space to the left and right
    ax1.legend(loc=0)
    ax1.tick_params(labelright=True)

    ax2 = plt.subplot2grid((4, 1), (2, 0))
    plt.boxplot(df[~df['num_orbits'].isnull()][attr], vert=False, widths=.8, sym='b|')
    ax2.yaxis.set_ticklabels([])
    ax2.set_xlim(*xlim)
    ax2.set_ylim(.4, 1.6)
    ax2.set_xlabel("Box-plot for all graphs")

    ax3 = plt.subplot2grid((4, 1), (3, 0))
    df2 = df[(df['n'] > df['num_orbits']) & ~df['num_orbits'].isnull()]
    df2.index = list(range(len(df2)))
    plt.boxplot(df2[attr], vert=False, widths=.8, sym='b|')
    ax3.yaxis.set_ticklabels([])
    ax3.set_xlim(*xlim)
    ax3.set_ylim(.4, 1.6)
    ax3.set_xlabel("Box-plot for symmetric graphs")

    fig.tight_layout()

    return fig, (ax1, ax2, ax3)


def create_histograms(df, args):
    bins = [0] + [10 ** i for i in range(0, 8)]

    # *** Histograms for n and m ***
    # axes = create_histogram_logx(df, 'n', bins=bins)
    fig, axes = create_histogram_logx(df, 'n', bins=bins)
    plt.tick_params(labelright=True)
    plt.savefig(os.path.join(args.target_dir, 'networkrepository_hist_n.pgf'))
    plt.close()

    bins = [0] + [10 ** i for i in range(0, 9)]
    # axes = create_histogram_logx(df, 'm', bins=bins)
    fig, axes = create_histogram_logx(df, 'm', bins=bins)
    plt.tick_params(labelright=True)
    plt.savefig(os.path.join(args.target_dir, 'networkrepository_hist_m.pgf'))
    plt.close()

    # *** Histogram for Modularity and density ***
    fig, axes = create_histogram_and_boxplot(df, 'modularity', attr_name='Modularity $Q$', xlabel='$Q$')
    plt.savefig(os.path.join(args.target_dir, 'networkrepository_hist_modularity.pgf'))
    plt.close()

    fig, axes = create_histogram_and_boxplot(df, 'density', attr_name='Density $\\rho$', xlabel='$\\rho$')
    plt.savefig(os.path.join(args.target_dir, 'networkrepository_hist_density.pgf'))
    plt.close()

    # *** Histogram for redundancy ***
    rng = (0, 1)
    num_bins = 20
    bins = [i / num_bins * (rng[1] - rng[0]) for i in range(num_bins+1)]

    fig = plt.figure()
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)

    plt.hist(df[~df['redundancy'].isnull()]['redundancy'], bins=bins, label='Network redundancy $r_G\'$')
    # plt.hist(df[~df['redundancy_m'].isnull()]['redundancy_m'], bins=bins, label='Network redundancy $r\'_G$')

    ax1.set_xlim(-.02, 1.02)  # Add space to the left and right
    # Get the number of asymmetric and transitive graphs
    no_asymmetric = len(df[df['redundancy'] == 0])
    no_transitive = len(df[df['redundancy'] == 1])
    ax1.set_xlabel("$r_G'$")
    # ax1.set_xlabel("$r\'_G$")
    ax1.set_ylabel("Frequency")
    # Add these numbers with bar markers
    plt.plot([0, 1], [no_asymmetric, no_transitive], 'b_')

    # Add ticks for those two values
    yticks = list(ax1.get_yticks())
    ax1.yaxis.set_ticks(yticks + [no_transitive, no_asymmetric])
    # Use the tick positions as labels, except for the two new ticks
    ax1.yaxis.set_ticklabels(list(map(int, yticks)) + ['', ''])

    # Add annotations for the asymmetric/transitive number of graphs
    arrowprops = {'color': 'black', 'arrowstyle': '-|>', 'relpos': (0.3, 0.5)}
    plt.annotate('Transitive graphs ({})'.format(no_transitive), xy=(1, no_transitive), xytext=(0.6, 4*no_transitive),
                 arrowprops=arrowprops, size=params["axes.labelsize"])
    arrowprops['relpos'] = (0.7, 0.5)
    plt.annotate('Asymmetric graphs ({})'.format(no_asymmetric), xy=(0, no_asymmetric), xytext=(0.1, 0.7*no_asymmetric),
                 arrowprops=arrowprops, size=params["axes.labelsize"])

    # Show tick labels on the right
    ax1.tick_params(labelright=True)

    ax2 = plt.subplot2grid((3, 1), (2, 0))
    plt.boxplot(df[~df['num_orbits'].isnull()]['redundancy'], vert=False, widths=.8, sym='b|')
    # plt.boxplot(df[~df['num_orbits'].isnull()]['redundancy_m'], vert=False, widths=.8, sym='b|')
    ax2.yaxis.set_ticklabels([])
    ax2.set_xlim(-.02, 1.02)
    ax2.set_ylim(.4, 1.6)
    ax2.set_xlabel("Box-plot for all graphs")

    fig.tight_layout()

    plt.savefig(os.path.join(args.target_dir, 'networkrepository_hist_redundancy.pgf'))
    # plt.savefig(os.path.join(args.target_dir, 'networkrepository_hist_redundancy_m.pgf'))
    plt.close()


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('infile_analysis', help='Path to the analysis csv file')
    argparser.add_argument('infile_stats', help='Path to the stats csv file')
    argparser.add_argument('target_dir',
                           help='Path to an (existing and writable!) directory to dump output',
                           type=PathType)
    argparser.add_argument('--keep_duplicates', help='Keep duplicates (identified by name)',
                           type=bool, default=False, nargs='?', const=True)

    args = argparser.parse_args()

    global DROP_DUPLICATES
    DROP_DUPLICATES = not args.keep_duplicates

    # =================================
    # Data loading
    # =================================
    df_a = pd.read_csv(args.infile_analysis, index_col=0)
    df_s = pd.read_csv(args.infile_stats, index_col=0)

    # Re-route the stdout so we can simply print our output
    orig_stdout = sys.stdout
    sys.stdout = io.StringIO()

    # =================================
    # Statistics on the networkrepository.com stats
    # =================================
    print_graph_stats_statistics(df_s)

    # =================================
    # Transformation
    # =================================
    df_a = transform_analysis_df(df_a)
    df_s = transform_stats_df(df_s)

    # Join both dataframes
    df = pd.merge(df_a, df_s, how='inner', left_on='name', right_on='Graph Name')

    df = transform_merged_df(df)

    # df = df[~df['Type'].isin(['bhoslib', 'dimacs'])]

    # =================================
    # Statistics on the whole data
    # =================================
    print_total_analysis_statistics(df.drop(['redundancy_m'], axis=1))

    # =================================
    # Statistics per type
    # =================================
    gb_type = df.drop(['redundancy_m'], axis=1).groupby(['Type'])
    for name, group in gb_type:
        print_group_analysis_statistics(name, group)
        # if len(group[~group['aut_group_size'].isnull()]) > 1:  # We need at least two data points
        #     scatter_matrix(group[['n', 'm', 'density', 'modularity', 'redundancy']], diagonal='kde')
        #     plt.show()
        # else:
        #     scatter_matrix(group[['n', 'm', 'density', 'modularity']], diagonal='kde')
        #     plt.show()

    # =================================
    # Restore stdout routing and save the results
    # =================================
    output = sys.stdout
    sys.stdout = orig_stdout
    output.seek(0)
    with open(os.path.join(args.target_dir, 'networkrepository_tables.tex'), 'w') as f:
        chunk = output.read(1024)
        while chunk:
            f.write(chunk)
            chunk = output.read(1024)

    # df.drop_duplicates(['name'], inplace=True)

    # =================================
    # Scatter plots
    # =================================
    create_scatterplots(df[~df['redundancy'].isnull()], args)

    # =================================
    # Histograms with box-plots
    # =================================
    create_histograms(df[~df['redundancy'].isnull()], args)


if __name__ == '__main__':
    main()
