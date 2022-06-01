#!/usr/bin/env python3

import sys

import numpy

def main():
    in_fname, out_fname = sys.argv[1:3]
    if in_fname.split('/')[-1].count('hg38') > 0:
        cellN = 15
    else:
        cellN = 11
    data = numpy.load(in_fname)
    print(data['chr'][:10])
    i = 0
    tads = []
    while i < data.shape[0]:
        chrom = data['chr'][i]
        TSS = data['TSS'][i]
        start = TSS
        end = TSS
        i += 1
        while i < data.shape[0] and data['TSS'][i] == TSS:
            start = min(start, data['cre-start'][i])
            end = max(start, data['cre-end'][i])
            i += 1
        where = numpy.where(data['cre-end'] == end)[0]
        if where.shape[0] > 0:
            i = where[-1]
            if data['TSS'][i] > end:
                end = data['TSS'][i]
            TSS = data['TSS'][i]
        tads.append([chrom, start - 1, end + 1])
        while i < data.shape[0] and data['TSS'][i] == TSS:
            i += 1
    output = open(out_fname, 'w')
    for line in tads:
        print(f"{line[0]}\t{line[1]}\t{line[2]}", file=output)
    output.close()

main()