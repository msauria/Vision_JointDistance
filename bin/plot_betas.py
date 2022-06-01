#!/usr/bin/env python3

import sys
import glob
import subprocess

import numpy
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_pdf import PdfPages


state_order = numpy.array([
    1, 0, 3, 2, 5, 9, 10, 4, 8, 12, 7, 18, 11, 24,
    13, 22, 21, 19, 20, 16, 17, 6, 15, 14, 23
    ], dtype=numpy.int32)

state_color = numpy.array([
    [226, 226, 226, 255], [247, 247, 247, 255], [153, 149, 242, 255],
    [169, 221, 173, 255], [236, 176, 215, 255], [248, 101, 246, 255],
    [247, 209, 147, 255], [246, 250, 126, 255], [ 40, 177,  58, 255],
    [247,   0,  16, 255], [ 75,  75,  75, 255], [ 10,   0, 195, 255],
    [  5,   0, 122, 255], [162,   0, 236, 255], [186,   0, 250, 255],
    [219,  12, 204, 255], [213,  24,  84, 255], [188,   0,  72, 255],
    [248, 128,   9, 255], [248, 197,  10, 255], [235, 238,   9, 255],
    [242, 247,  12, 255], [244,   0,   6, 255], [244,   0,   6, 255],
    [237,   0,   5, 255]
    ], dtype=numpy.float32) / 255

stateN = state_order.shape[0]

def main():
    in_fname, out_fname = sys.argv[1:3]
    dtypes = [('Condition', "<U7"), ("Genome", "<U4"), ('Features', "<U8"),
              ("CT", "<U30"), ("Rep", numpy.int32)]
    for i in range(stateN):
        dtypes.append((f"P{i}", numpy.float32))
    for i in range(stateN):
        dtypes.append((f"C{i}", numpy.float32))
    data = numpy.loadtxt(in_fname, dtype=numpy.dtype(dtypes), skiprows=1)
    CTs = [("mm10", x) for x in numpy.unique(data['CT'][numpy.where(data['Genome'] == 'mm10')])]
    CTs += [("hg38", x) for x in numpy.unique(data['CT'][numpy.where(data['Genome'] == 'hg38')])]
    cellN = len(CTs)
    fig, ax = plt.subplots(3, 4, figsize=(20, 11),
                           gridspec_kw={'height_ratios': (1, cellN, cellN),
                                        #'left': 0.2, 
                                        #'right': 0.99,
                                        #'top': 0.95, 'bottom': 0.0,
                                        'hspace':0
                                        })
    for i, f in enumerate(['promoter both', 'cre both', 'promoter', 'cre']):
        for j, c in enumerate(['treat', 'control']):
            plot_betas(data, ax[j + 1, i], f, c, CTs)
    cmap = ListedColormap(state_color)
    for i, l in enumerate(['Promoter (both)', 'CRE (both)', 'Promoter', 'CRE']):
        ax[0, i].imshow(numpy.arange(stateN).reshape(1, -1), cmap=cmap)
        ax[0, i].set_xticks(numpy.arange(stateN))
        ax[0, i].set_xticklabels([str(x) for x in state_order])
        ax[0, i].get_yaxis().set_visible(False)
        #ax[0, i].set_xlabel(l, fontsize=18)
        ax[0, i].text(stateN / 2, -1, l, horizontalalignment='center', fontsize=18)
        if i == 0:
            ax[0, i].margins(tight=True)
        else:
            ax[0, i].get_yaxis().set_visible(False)
    #ax[1, 0].text(-8, len(CTs) / 2, "Treat", rotation=90, fontsize=18)
    #ax[2, 0].text(-8, len(CTs) / 2, "Control", rotation=90, fontsize=18)
    plt.tight_layout()
    fig.savefig(out_fname)

def plot_betas(alldata, ax, feat, cond, CTs):
    minval = numpy.inf
    maxval = -numpy.inf
    if feat.split()[0] == 'promoter':
        prefix = 'P'
    else:
        prefix = 'C'
    f = feat.split()[0]
    data = alldata[numpy.where((alldata['Rep'] == 0) &
                               ((alldata['Features'] == f) |
                                (alldata['Features'] == 'both')))]
    for i in range(stateN):
        minval = min(minval, numpy.amin(data[f"{prefix}{i}"]))
        maxval = max(maxval, numpy.amax(data[f"{prefix}{i}"]))

    f = feat.split()[-1]
    data = alldata[numpy.where((alldata['Condition'] == cond) &
                               (alldata['Rep'] == 0) &
                               (alldata['Features'] == f))]
    hm = numpy.zeros((stateN, len(CTs)), dtype=numpy.float32)
    for i, ct in enumerate(CTs):
        temp = data[numpy.where((data['CT'] == ct[1]) & (data['Genome'] == ct[0]))]
        for j, k in enumerate(state_order):
            hm[j, i] = temp[f"{prefix}{k}"]
    cmap = cm.seismic
    maxval = max(maxval, -minval)
    ax.imshow(hm.T, cmap=cmap, norm=colors.Normalize(vmin=-maxval, vmax=maxval))
    ax.get_xaxis().set_visible(False)
    if feat == 'promoter both':
        ax.margins(x=6, tight=True)
        ax.set_yticks(numpy.arange(hm.shape[1]))
        ax.set_yticklabels([f"{x[0]} {x[1]}" for x in CTs])
        ax.set_ylabel(cond, fontsize=18)
    else:
        ax.get_yaxis().set_visible(False)


main()
