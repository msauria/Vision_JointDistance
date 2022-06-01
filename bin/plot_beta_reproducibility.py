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
              ("CT", "<U30"), ("Rep", numpy.int32),
              ('pBetas', numpy.float32, (stateN,)),
              ('cBetas', numpy.float32, (stateN,))]
    data = numpy.loadtxt(in_fname, dtype=numpy.dtype(dtypes), skiprows=1)
    data = data[numpy.where(data['Condition'] == 'treat')]
    data['pBetas'] = data['pBetas'][:, state_order]
    data['cBetas'] = data['cBetas'][:, state_order]
    fig, allax = plt.subplots(3, 2, figsize=(12, 8))
    for i, f in enumerate(['promoter both', 'cre both', 'cre']):
        ymin = 0
        ymax = 0
        for j, g in enumerate(['mm10', 'hg38']):                
            temp = plot_betas(data, allax[i, j], f, g)
            ymin = min(ymin, temp[0])
            ymax = max(ymax, temp[1])
        for j in range(2):
            allax[i, j].set_ylim(ymin, ymax)
            #allax[i, j].set_ylim(-300, 300)
    plt.tight_layout()
    fig.savefig(out_fname)

def plot_betas(alldata, ax, feat, genome):
    cmap = ListedColormap(state_color)
    pcolors = []
    for i in range(stateN):
        pcolors.append(cmap((i + 0.5) / stateN))
    f = feat.split()[-1]
    if feat.split()[0] == 'promoter':
        prefix = 'p'
        ax.set_title(genome)
    else:
        prefix = "c"
    data = alldata[numpy.where((alldata['Genome'] == genome) &
                               (alldata['Features'] == f))]
    nzdata = data[numpy.where(data['Rep'] > 0)]
    zdata = data[numpy.where(data['Rep'] == 0)]
    CTs = numpy.unique(data['CT'])
    X = []
    Y = []
    Y2 = []
    X2 = []
    pcolors = []
    pcolors2 = []
    for i, ct in enumerate(CTs):
        nzd = numpy.copy(nzdata[numpy.where(nzdata['CT'] == ct)])
        zd = zdata[numpy.where(zdata['CT'] == ct)]
        #nzd[f"{prefix}Betas"] -= zd[f"{prefix}Betas"].reshape(1, -1)
        #nzd[f"{prefix}Betas"] /= zd[f"{prefix}Betas"].reshape(1, -1)
        N = nzd.shape[0]
        for j in range(stateN):
            X += [j + 0.5] * N
            Y += list(nzd[f"{prefix}Betas"][:, j])
            pcolors += [cmap((j + 0.5) / stateN)] * N
            X2.append(j + 0.5)
            Y2.append(zd[f"{prefix}Betas"][0, j])
            pcolors2.append(cmap((j + 0.5) / stateN))
    ymin = min(numpy.amin(numpy.array(Y)), numpy.amin(numpy.array(Y2))) 
    ymax = max(numpy.amax(numpy.array(Y)), numpy.amax(numpy.array(Y2))) 
    ax.scatter(X, Y, color=pcolors)
    ax.scatter(X2, Y2, color=pcolors2, edgecolors='black')
    ax.set_xlim(0, stateN)
    ax.set_ylabel('Beta value')
    ax.set_xlabel(feat)
    ax.set_xticks(numpy.arange(stateN) + 0.5)
    ax.set_xticklabels([f"{x}" for x in numpy.arange(stateN)])
    return ymin, ymax



main()