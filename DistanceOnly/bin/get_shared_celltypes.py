#!/usr/bin/env python3

import sys

import numpy


def main():
    fname1, fname2 = sys.argv[1:3]
    temp = numpy.load(fname1)
    names1 = set([x.split('_')[0] for x in temp.dtype.names[3:]])
    temp = numpy.load(fname2)
    names2 = set([x.split('_')[0] for x in temp.dtype.names[3:]])
    names = names1.intersection(names2)
    names = list(names)
    names.sort()
    print(" ".join(names))

if __name__ == "__main__":
    main()