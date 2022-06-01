#!/usr/bin/env python3

import sys
import glob

def main():
    out_fname = sys.argv[1]
    results = []
    for genome in ['mm10', 'hg38']:
        dirs = glob.glob(f'Results_{genome}/*')
        CTs = [x.split('_')[-1] for x in dirs]
        for ct in CTs:
            fnames = glob.glob(f'Results_{genome}/{genome}_{ct}/*correlations.txt')
            for fname in fnames:
                if fname.count('control') > 0:
                    cond = 'control'
                else:
                    cond = 'treat'
                rep = fname.split('/')[-1].split('_')[0]
                feat = fname.split('_')[-2]
                for line in open(fname):
                    line = line.rstrip().split()
                    if line[0] == genome and line[1] == ct:
                        results.append([cond, genome, feat, ct, rep, line[2]])
    results.sort()
    output = open(out_fname, 'w')
    print("Condition\tGenome\tFeatures\tCT\tRep\tR2", file=output)
    for line in results:
        print("\t".join(line), file=output)
    output.close()

main()