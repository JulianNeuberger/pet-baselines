import dataclasses
import math
import os
import sqlite3
import typing

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

sns.set_theme(rc={
    'figure.autolayout': False,
    'font.family': ['Computer Modern', 'CMU Serif', 'cmu', 'serif'],
    'font.serif': ['Computer Modern', 'CMU Serif', 'cmu'],
    # 'text.usetex': True
})
matplotlib.rcParams.update({
    'figure.autolayout': False,
    'font.family': ['Computer Modern', 'CMU Serif', 'cmu', 'serif'],
    'font.serif': ['Computer Modern', 'CMU Serif', 'cmu'],
    # 'text.usetex': True
})
sns.set_style(rc={
    'font.family': ['Computer Modern', 'CMU Serif', 'cmu', 'serif'],
    'font.serif': ['Computer Modern', 'CMU Serif', 'cmu'],
    # 'text.usetex': True
})
sns.set(font='Computer Modern')


@dataclasses.dataclass
class PlottingData:
    data: pd.DataFrame
    x: str
    y: str
    hue: str
    type: str


def radar_factory(num_vars, frame='circle'):
    """
    Directly taken from https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = f'{num_vars}-spoke-radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
            return lines

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def plot_co_ref_hyper_opt():
    # Read sqlite query results into a pandas DataFrame
    con = sqlite3.connect('optuna.db')
    df: pd.DataFrame
    df = pd.read_sql_query('SELECT t.trial_id, value, param_name, param_value '
                           'FROM trial_values '
                           'LEFT JOIN trials t ON t.trial_id = trial_values.trial_id and study_id=1 '
                           'LEFT JOIN trial_params tp on t.trial_id = tp.trial_id', con)

    trial_value_df: pandas.DataFrame = df.groupby('trial_id').first()['value']
    parameter_df = df.pivot_table(index='trial_id', columns='param_name', values='param_value', aggfunc='first')

    df = parameter_df.join(trial_value_df, on='trial_id')
    df = df.pivot_table(index='cluster_overlap', columns='mention_overlap', values='value')
    df = df[[c for c in df.columns if float(c) > 0]]

    x = df.index.values
    y = df.columns.values
    z = df.values

    X, Y = np.meshgrid(x, y)
    Z = df.T

    sns.set_theme()

    levels = np.linspace(.2, .5, num=15)

    cs = plt.contourf(X, Y, Z, levels=levels)
    plt.xlabel(r'$\alpha_c$')
    plt.ylabel(r'$\alpha_m$')
    ticks = [f'{l:.1%}' for l in levels]
    cbar = plt.colorbar(cs, ticks=levels)
    cbar.ax.set_xticklabels(ticks)
    cbar.ax.set_ylabel('F1')

    os.makedirs(os.path.join('figures', 'hyper-opt'), exist_ok=True)
    plt.savefig(os.path.join('figures', 'hyper-opt', 'co-ref-params.pdf'))
    plt.savefig(os.path.join('figures', 'hyper-opt', 'co-ref-params.png'))


def place_text_at_spoke(ax, theta: float, r: float, text: str,
                        color='black', font_size: float = 12,
                        offset: float = .05, rotate: bool = False):
    ha = 'center'
    va = 'center'
    angle = theta

    if not rotate:
        if angle != 0 and angle != math.pi:
            if angle > math.pi:
                ha = 'left'
            elif angle < math.pi:
                ha = 'right'

        if angle == 0:
            va = 'bottom'
        elif abs(angle - math.pi) < 1e-5:
            va = 'top'

    rotation = 0.0
    if rotate:
        rotation = math.degrees(theta)
        if theta > math.pi / 2:
            rotation += 180
        if theta > 3 * math.pi / 2:
            rotation += 180

    ax.text(theta, r + offset, text,
            color=color,
            rotation=rotation,
            fontsize=font_size,
            fontfamily='serif',
            horizontalalignment=ha,
            verticalalignment=va)


def make_spider(df: pd.DataFrame, grid_spec, title: str):
    assert len(df) > 0, f'Empty df for title "{title}"'

    categories = list(df)[0:]
    num_categories = len(categories)

    theta = radar_factory(num_categories, frame=f'polygon')
    ax = plt.subplot(grid_spec, projection=f'{num_categories}-spoke-radar')

    ax.set_yticklabels([])

    plt.ylim(0, 1)

    spoke_labels = df.columns.values
    ax.set_varlabels(['' for _ in spoke_labels])

    handles = []
    max_for_tag = {}
    for row_index, row in enumerate(df.index):
        experiment_data = df.loc[row].values.tolist()
        label = label_from_experiment_name(row)
        handle, = ax.plot(theta, experiment_data, linewidth=1, linestyle='solid', label=label)
        handles.append(handle)
        ax.fill(theta, experiment_data, alpha=.2)
        for i, value in enumerate(experiment_data):
            if i not in max_for_tag:
                max_for_tag[i] = value, row_index
            if max_for_tag[i][0] < value:
                max_for_tag[i] = value, row_index

    for i, (value, row_index) in max_for_tag.items():
        place_text_at_spoke(ax, theta[i], value, f'{value:.2f}', color=sns.color_palette()[row_index], font_size=13)

    for i, spoke_label in enumerate(spoke_labels):
        offset = 0.1
        label_rows = spoke_label.split(' ')
        num_label_rows = len(label_rows)
        offset += .1 * (num_label_rows - 1)
        if max_for_tag[i][0] > .75:
            offset += (.25 - (1.0 - max_for_tag[i][0])) * .5
        spoke_label = '\n'.join(label_rows)
        place_text_at_spoke(ax, theta[i], 1.0, spoke_label, offset=offset, rotate=True, font_size=13)

    ax.set_title(title, loc='left', pad=30, fontsize=14, fontfamily='serif')

    return handles


def label_from_experiment_name(experiment_name: str) -> str:
    mapping = {
        'cat-boost': 'BoostRelEx',
        'rule-based': 'RuleRelEx',
        'jerex': 'Jerex',
        'neural-coref': 'neural ER',
        'naive-coref': 'naive ER'
    }

    for k, v in mapping.items():
        if k in experiment_name:
            return v
    raise ValueError(f'Unknown approach in experiment name {experiment_name}')


def make_bars(df: pd.DataFrame, grid_spec, title: str, x: str, y: str, hue: str):
    ax = plt.subplot(grid_spec)

    for experiment_name in df['experiment_name'].values:
        df = df.replace(experiment_name, label_from_experiment_name(experiment_name))

    sns.barplot(df, x=x, y=y, hue=hue)

    ax.xaxis.label.set_visible(False)
    ax.set_ylim((0, 1.0))
    ax.set_ylabel(ax.get_ylabel(), fontfamily='serif')
    ax.set_xlabel(ax.get_xlabel(), fontfamily='serif')

    for label in ax.get_xticklabels():
        label.set_fontproperties('serif')

    for label in ax.get_yticklabels():
        label.set_fontproperties('serif')

    ax.set_title(title, loc='left', pad=30, fontsize=14, fontfamily='serif')
    handles, _ = ax.get_legend_handles_labels()

    for container in handles:
        for bar in container:
            base_color = bar.get_facecolor()
            edge_color = base_color
            face_color = (*base_color[:3], .5)

            bar.set_facecolor(face_color)
            bar.set_edgecolor(edge_color)

    ax.get_legend().remove()
    return handles


def plot_experiment_results(grouped_data: typing.List[PlottingData],
                            name: str, legends_at: typing.Dict[int, int],
                            widths: typing.List[float] = None, margins: typing.Dict[str, float] = None):
    fig: plt.Figure
    fig = plt.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(2, len(grouped_data), height_ratios=[1, 5], width_ratios=widths)

    sns.set_theme()

    handles = []
    for i, plotting_data in enumerate(grouped_data):
        # data = df.filter(items=experiment_names, axis=0).dropna(axis=1)
        # print(data)
        if plotting_data.type == 'spider':
            _handles = make_spider(plotting_data.data, grid_spec[i + len(grouped_data)], title=f'{chr(i + 97)})')
        else:
            _handles = make_bars(plotting_data.data, grid_spec[i + len(grouped_data)],
                                 title=f'{chr(i + 97)})',
                                 x=plotting_data.x, y=plotting_data.y, hue=plotting_data.hue)
        handles.append(_handles)

    assert len(handles) > 0

    for legend_to, legend_from in legends_at.items():
        legend_axis = fig.add_subplot(int(f'2{len(grouped_data)}{legend_to + 1}'))
        legend_axis.axis('off')
        legend_axis.legend(handles=handles[legend_from], borderaxespad=0, ncol=len(handles[legend_from]),
                           prop={'family': 'serif'},
                           bbox_to_anchor=(1, 1), loc='upper right')

    if margins is not None:
        plt.subplots_adjust(**margins)

    os.makedirs('figures/results', exist_ok=True)
    plt.savefig(f'figures/results/{name}.png', bbox_inches='tight')
    plt.savefig(f'figures/results/{name}.pdf', bbox_inches='tight')


def plot_scenario_4_5_6():
    df: pd.DataFrame = pd.read_pickle('experiments.jerex.pkl')
    df = df.reset_index()
    df = df.loc[df['tag'] != 'overall']
    df = df[['experiment_name', 'tag', 'f1']]
    df = df.pivot_table(values='f1', index='experiment_name', columns='tag')

    grouped_data = [
        PlottingData(data=df.filter(items=experiment_names, axis=0).dropna(axis=1),
                     hue='experiment_name', x='tag', y='f1', type='spider')
        for experiment_names in [['complete-cat-boost', 'complete-rule-based', 'jerex-relations'],
                                 ['co-ref-only-cat-boost', 'co-ref-only-rule-based'],
                                 ['cat-boost-isolated', 'rule-based-isolated']]
    ]

    plot_experiment_results(grouped_data, name='scenario-4-5-6', legends_at={2: 0})


def plot_scenario_1_2_3():
    grouped_data = []

    for experiment_names in [['cat-boost-isolated', 'rule-based-isolated'],
                             ['complete-cat-boost', 'complete-rule-based', 'jerex-relations']]:
        df: pd.DataFrame = pd.read_pickle('experiments.jerex.pkl')
        df = df.reset_index()
        df = df.loc[df['tag'] == 'overall']
        df = df[['experiment_name', 'f1', 'p', 'r']]
        df = df.melt(id_vars=['experiment_name'], var_name='metric', value_name='score')
        df = df[df['experiment_name'].isin(experiment_names)]
        grouped_data.append(PlottingData(data=df, hue='experiment_name', x='metric', y='score', type='bar'))

    for experiment_names in [['neural-coref', 'naive-coref'],
                             ['neural-coref-perfect-mentions', 'naive-coref-perfect-mentions']]:
        df: pd.DataFrame = pd.read_pickle('experiments.jerex.pkl')
        df = df.reset_index()
        df = df.loc[df['tag'] != 'overall']
        df = df[['experiment_name', 'tag', 'f1']]
        df = df.pivot_table(values='f1', index='experiment_name', columns='tag')
        data = df.filter(items=experiment_names, axis=0)
        data = data.dropna(axis=1)
        data = data.reset_index()
        data = data.melt(id_vars='experiment_name', value_vars=list(data.columns[1:]), var_name='tag',
                         value_name='f1')
        grouped_data.append(PlottingData(data=data, hue='experiment_name', x='tag', y='f1', type='bar'))

    plot_experiment_results(grouped_data, name='scenario-1-2-3',
                            legends_at={1: 1, 3: 3},
                            widths=[1.25, 1.25, .9, .9],
                            margins={'wspace': .3, 'hspace': 0.15})


if __name__ == '__main__':
    plot_scenario_4_5_6()
    plot_scenario_1_2_3()
