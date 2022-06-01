#!/usr/bin/env python3

import sys

import numpy
import matplotlib.pyplot as plt
from matplotlib import cm, colors, markers
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_pdf import PdfPages


def main():
    in_fname, out_fname = sys.argv[1:3]
    dtypes = [('Condition', "<U7"), ("Genome", "<U4"), ('Features', "<U8"),
              ("CT", "<U30"), ("Rep", numpy.int32), ("R2", numpy.float32)]
    alldata = numpy.loadtxt(in_fname, dtype=numpy.dtype(dtypes), skiprows=1)
    alldata = alldata[numpy.where(alldata['Condition'] == 'treat')]
    fig, all_ax = plt.subplots(2, 2, figsize=(12,6),
                               gridspec_kw={'height_ratios': (7, 1)})
    minval = numpy.inf
    maxval = -numpy.inf
    for h, feat in enumerate([('both', "Promoters + CREs"), ('cre', "CREs")]):
        ax = all_ax[0, h]
        ax.set_title(feat[1])
        all_CTs = numpy.unique(alldata['CT'])
        cmap = cm.get_cmap('tab20')
        color_dict = {}
        for i, ct in enumerate(all_CTs):
            color_dict[ct] = cmap((i + 0.5) / all_CTs.shape[0])
        shape_dict = {'mm10': "o", 'hg38': "D"}
        for g in ['mm10', 'hg38']:
            data = alldata[numpy.where((alldata['Genome'] == g) &
                                       (alldata['Features'] == feat[0]))]
            nzdata = data[numpy.where(data['Rep'] > 0)]
            zdata = data[numpy.where(data['Rep'] == 0)]
            minval = min(minval, numpy.amin(zdata['R2']), numpy.amin(nzdata['R2']))
            maxval = max(maxval, numpy.amax(zdata['R2']), numpy.amax(nzdata['R2']))
            CTs = numpy.unique(data['CT'])
            shape = shape_dict[g]
            for ct in CTs:
                nzd = nzdata[numpy.where(nzdata['CT'] == ct)]
                zd = zdata[numpy.where(zdata['CT'] == ct)]
                pcolor = color_dict[ct]
                ax.scatter([zd['R2'][0]] * nzd.shape[0], nzd['R2'],
                           color=pcolor, marker=shape)
                print(zd['R2'][0], list(nzd['R2']))
        ax.set_xlabel(r'Adjusted R^2, fixed initialization')
        ax.set_ylabel(r'Adjusted R^2, random initialization')
    span = maxval - minval
    minval -= 0.05 * span
    maxval += 0.05 * span
    for i in range(2):
        all_ax[0, i].set_xlim(minval, maxval)
        all_ax[0, i].set_ylim(minval, maxval)
    all_ax[1, 1].axis('off')
    ax = all_ax[1, 0]
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 3)
    ax.scatter([0.1], [2], marker='o')
    ax.annotate('mm10', (0.17, 1.75))
    ax.scatter([0.1], [1], marker='D')
    ax.annotate('hg38', (0.17, 0.75))
    X = numpy.arange(all_CTs.shape[0]) // 3 + 1
    Y = numpy.arange(all_CTs.shape[0]) % 3 + 0.5
    pcolors = []
    for ct in all_CTs:
        pcolors.append(color_dict[ct])
    ax.scatter(X, Y, marker='s', color=pcolors)
    for i, txt in enumerate(all_CTs):
        ax.annotate(txt, (X[i] + 0.07, Y[i] - 0.25), color=pcolors[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    fig.savefig(out_fname)


main()