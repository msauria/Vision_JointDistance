#!/usr/bin/env python3

import sys
import argparse

import numpy


def main():
    parser = generate_parser()
    args = parser.parse_args()
    print("\r%s\rLoading data" % (' ' * 80), end='', file=sys.stderr)
    fs = open(args.STATE, 'r')
    header = fs.readline().rstrip('\n\r').split()[:-1]
    data = []
    states = []
    chromlen = 0
    for line in fs:
        line = line.rstrip('\r\n').split()
        binID, chrom, start, end = line[:4]
        state = tuple(line[4:-1])
        data.append([chrom, int(start), int(end)])
        states.append(state)
        chromlen = max(chromlen, len(chrom))
    states = numpy.array(states, dtype=numpy.int8)
    dtype = [('chr', 'U%i' % chromlen), ('start', numpy.int32), ('end', numpy.int32)]
    celltypes = {}
    ct_names = []
    for name in header[4:]:
        ct = name.split("_")[0]
        celltypes.setdefault(ct, 0)
        celltypes[ct] += 1
        ct_names.append("{}_{}".format(ct, celltypes[ct]))
    print(ct_names)
    order = numpy.argsort(numpy.array(ct_names))
    print(order)
    states = states[:, order]
    for i in range(len(data)):
        data[i] = tuple(data[i] + list(states[i, :]))
    dtype += [(ct_names[x], numpy.int8) for x in order]
    data = numpy.array(data, dtype=numpy.dtype(dtype))
    # Remove bins with all zero states
    print("\r%s\rRemoving zero bins" % (' ' * 80), end='', file=sys.stderr)
    maxState = numpy.zeros(data.shape[0])
    names = data.dtype.names[3:]
    cellN = len(names)
    for i in range(cellN):
        maxState = numpy.maximum(maxState, data[names[i]])
        stateN = numpy.amax(data[names[i]]) + 1
    data = data[numpy.where(maxState > 0)]
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
    parser.add_argument('-s', '--state', dest="STATE", type=str, action='store', required=True,
                        help="State text file")
    parser.add_argument('-o', '--output', dest="OUTPUT", type=str, action='store', required=True,
                        help="State npy file")
    return parser


if __name__ == "__main__":
    main()
