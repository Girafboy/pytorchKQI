#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' a module for figure '

__author__ = 'Huquan Kang'

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


def initialize(width=3.5, height=3.5, left=False, bottom=False, right=False, top=False, left_tick=False, bottom_tick=False, right_tick=False, top_tick=False, autolayout=False, serif=False):
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['figure.autolayout'] = autolayout
    plt.rcParams['figure.subplot.left'] = 0
    plt.rcParams['figure.subplot.right'] = 1
    plt.rcParams['figure.subplot.bottom'] = 0
    plt.rcParams['figure.subplot.top'] = 1

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.family'] = 'serif' if serif else 'sans-serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['font.size'] = 7
    plt.rcParams['axes.titlesize'] = 7
    plt.rcParams['axes.labelsize'] = 7
    plt.rcParams['xtick.labelsize'] = 7
    plt.rcParams['ytick.labelsize'] = 7
    plt.rcParams['legend.fontsize'] = 7

    plt.rcParams['lines.linewidth'] = 1.0

    plt.rcParams['figure.frameon'] = False
    plt.rcParams['legend.frameon'] = False

    plt.rcParams['figure.figsize'] = [width, height]  # 3.5 (single column) / 7.2 (double column) * 9.7 (full depth)
    plt.rcParams['xtick.top'] = top_tick
    plt.rcParams['xtick.bottom'] = bottom_tick
    plt.rcParams['xtick.labeltop'] = top_tick
    plt.rcParams['xtick.labelbottom'] = bottom_tick
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['xtick.major.size'] = 3.0
    plt.rcParams['xtick.minor.size'] = 1.0
    plt.rcParams['xtick.minor.visible'] = False
    plt.rcParams['ytick.left'] = left_tick
    plt.rcParams['ytick.right'] = right_tick
    plt.rcParams['ytick.labelleft'] = left_tick
    plt.rcParams['ytick.labelright'] = right_tick
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['ytick.major.size'] = 3.0
    plt.rcParams['ytick.minor.size'] = 1.0
    plt.rcParams['ytick.minor.visible'] = False
    plt.rcParams['axes.ymargin'] = .01
    plt.rcParams['axes.spines.left'] = left
    plt.rcParams['axes.spines.bottom'] = bottom
    plt.rcParams['axes.spines.top'] = top
    plt.rcParams['axes.spines.right'] = right
    plt.rcParams['axes.formatter.limits'] = (-3, 4)
    plt.rcParams['axes.formatter.use_locale'] = True

    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.transparent'] = True
    plt.rcParams['savefig.pad_inches'] = 0


def gradient_colors(color_low='#2AB179', color_high='#C45351', num_colors=10):
    cmap = LinearSegmentedColormap.from_list('gradient', [(0, color_low), (1, color_high)])
    return [cmap(i) for i in np.linspace(0, 1, num_colors)]
