#!/usr/bin/env python3

import sys
import argparse

import numpy


def main():
    parser = generate_parser()
    args = parser.parse_args()
    print("\r%s\rLoading data" % (' ' * 80), end='', file=sys.stderr)
    fs = open(args.CRE, 'r')
    data = []
    chromlen = 0
    for line in fs:
        line = line.rstrip('\r\n').split()
        chrom, start, end = line[:3]
        try:
            data.append(tuple([chrom, int(start), int(end)]))
        except:
            pass
        chromlen = max(chromlen, len(chrom))
    dtype = [('chr', 'U%i' % chromlen), ('start', numpy.int32), ('end', numpy.int32)]
    data = numpy.array(data, dtype=numpy.dtype(dtype))
    # Sort data
    print("\r%s\rSorting data" % (' ' * 80), end='', file=sys.stderr)
    data = data[numpy.lexsort((data['start'], data['chr']))]
    print("\r%s\rSaving data" % (' ' * 80), end='', file=sys.stderr)
    numpy.save(args.OUTPUT, data)
    print("\r%s\r" % (' ' * 80), end='', file=sys.stderr)


def generate_parser():
    """Generate an argument parser."""
    description = "%(prog)s -- Convert an IDEAS state text file to a numpy NPY file"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-c', '--cre', dest="CRE", type=str, action='store', required=True,
                        help="CRE text file")
    parser.add_argument('-o', '--output', dest="OUTPUT", type=str, action='store', required=True,
                        help="CRE npy file")
    return parser


if __name__ == "__main__":
    main()
