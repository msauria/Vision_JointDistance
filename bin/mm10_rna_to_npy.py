#!/usr/bin/env python3

import sys
import argparse

import numpy


def main():
    parser = generate_parser()
    args = parser.parse_args()
    fs = open(args.RNA, 'r')
    CTs = ['CFUE', 'CFUMk', 'CMP', 'ERY', 'GMP', 'iMK',
           'LSK', 'MEP', 'MON', 'NEU', 'ER4', 'G1E']
    ct_names = fs.readline().rstrip('\r\n').split()[6:]
    for i in range(len(ct_names)):
        for ct in CTs:
            if ct_names[i][:len(ct)] == ct:
                ct_names[i] = "{}_{}".format(ct, ct_names[i][len(ct):])
                break
    chromlen = 0
    strand2bool = {'+': False, '-': True}
    data = []
    rna = []
    for line in fs:
        line = line.rstrip('\r\n').split()
        chrom, start, end, gene, gene_type, strand = line[:6]
        if gene_type != 'protein_coding':
            continue
        TPM = numpy.array(line[6:], dtype=numpy.float32)
        strand = strand2bool[strand]
        if strand:
            tss = end
        else:
            tss = start
        data.append([chrom, int(tss), strand])
        rna.append(line[6:])
        chromlen = max(chromlen, len(chrom))
    fs.close()
    rna = numpy.array(rna, dtype=numpy.float32)
    dtype = [('chr', 'U%i' % chromlen), ('TSS', numpy.int32), ('strand', bool)]
    celltypes = {}
    print(numpy.amin(rna), numpy.mean(rna), numpy.amax(rna))
    rna /= numpy.sum(rna, axis=0, keepdims=True) / 5000000
    rna = numpy.log2(rna + 1)
    for i in range(len(data)):
        data[i] = tuple(data[i] + list(rna[i, :]))
    dtype += [(x, numpy.float32) for x in ct_names]
    data = numpy.array(data, dtype=numpy.dtype(dtype))
    data = data[numpy.lexsort((data['TSS'], data['chr']))]
    old_len = data.shape[0]
    unique = numpy.r_[True, numpy.logical_or(data['chr'][1:] != data['chr'][:-1],
                                             data['TSS'][1:] != data['TSS'][:-1])]
    data = data[unique]
    print("Old #: {}  New #: {}".format(old_len, data.shape[0]))
    numpy.save(args.OUTPUT, data)


def generate_parser():
    """Generate an argument parser."""
    description = "%(prog)s -- Convert an RNA TPM text file to a numpy NPY file"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-r', '--rna', dest="RNA", type=str, action='store', required=True,
                        help="RNA expression file")
    parser.add_argument('-o', '--output', dest="OUTPUT", type=str, action='store', required=True,
                        help="Numpy NPY RNA expression file to write to")
    return parser


if __name__ == "__main__":
    main()
