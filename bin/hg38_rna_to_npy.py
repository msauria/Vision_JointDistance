#!/usr/bin/env python3

import sys
import argparse
import gzip

import numpy


def main():
    parser = generate_parser()
    args = parser.parse_args()
    IDs = load_IDs(args.IDS)
    print(len(IDs))
    fs = open(args.RNA, 'r')
    ct_names = fs.readline().rstrip('\r\n').split()[1:]
    chromlen = 0
    strand2bool = {'+': False, '-': True}
    data = []
    rna = []
    missing = 0
    for line in fs:
        line = line.rstrip('\r\n').split()
        gene = line[0].split('.')[0]
        if gene not in IDs:
            missing += 1
            continue
        chrom, tss, strand = IDs[gene]
        data.append([chrom, int(tss), strand])
        rna.append(line[1:])
        chromlen = max(chromlen, len(chrom))
    fs.close()
    rna = numpy.array(rna, dtype=numpy.float32)
    print(numpy.amin(rna), numpy.mean(rna), numpy.amax(rna))
    rna /= numpy.sum(rna, axis=0, keepdims=True) / 5000000
    print(numpy.amin(rna), numpy.mean(rna), numpy.amax(rna))
    rna = numpy.log2(rna + 1)
    print("Missing IDs: {}\n".format(missing))
    dtype = [('chr', 'U%i' % chromlen), ('TSS', numpy.int32), ('strand', bool)]
    celltypes = {}
    for i in range(len(data)):
        data[i] = tuple(data[i] + list(rna[i, :]))
    dtype += [(x, numpy.float32) for x in ct_names]
    data = numpy.array(data, dtype=numpy.dtype(dtype))
    data = data[numpy.lexsort((data['TSS'], data['chr']))]
    unique = numpy.r_[True, numpy.logical_or(data['chr'][1:] != data['chr'][:-1],
                                             data['TSS'][1:] != data['TSS'][:-1])]
    print(data.shape[0], unique.shape[0])
    data = data[unique]
    numpy.save(args.OUTPUT, data)

def load_IDs(fname):
    IDs = {}
    for line in open(fname):
        line = line.rstrip().split("\t")
        if line[4] != "protein_coding":
            continue
        if not line[6].isdigit() and line[6] != "X":
            continue
        gene = line[0]
        if line[3] == "1":
            if gene not in IDs or IDs[gene][1] > int(line[1]):
                IDs[gene] = ("chr{}".format(line[6]), int(line[1]), False)
        else:
            if gene not in IDs or IDs[gene][1] < int(line[2]):
                IDs[gene] = ("chr{}".format(line[6]), int(line[2]), False)
    return IDs

def generate_parser():
    """Generate an argument parser."""
    description = "%(prog)s -- Convert an RNA TPM text file to a numpy NPY file"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-r', '--rna', dest="RNA", type=str, action='store', required=True,
                        help="RNA expression file")
    parser.add_argument('-i', '--ids', dest="IDS", type=str, action='store', required=True,
                        help="ENSEMBL ID file")
    parser.add_argument('-o', '--output', dest="OUTPUT", type=str, action='store', required=True,
                        help="Numpy NPY RNA expression file to write to")
    return parser


if __name__ == "__main__":
    main()
