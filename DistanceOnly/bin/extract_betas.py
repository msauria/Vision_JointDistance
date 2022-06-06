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
            fnames = glob.glob(f'Results_{genome}/{genome}_{ct}/*betas.txt')
            for fname in fnames:
                if fname.count('control') > 0:
                    cond = 'control'
                else:
                    cond = 'treat'
                rep = fname.split('/')[-1].split('_')[0]
                feat = fname.split('_')[-2]
                promoter = []
                cre = []
                for line in open(fname):
                    line = line.rstrip().split()
                    if line[0] == 'Promoter':
                        promoter.append(line[2])
                    elif line[0] == "CRE":
                        cre.append(line[2])
                if len(cre) == 0:
                    cre = ['0'] * len(promoter)
                if len(promoter) == 0:
                    promoter = ['0'] * len(cre)
                results.append([cond, genome, feat, ct, rep] + promoter + cre)
    results.sort()
    n = (len(results[0]) - 5) // 2
    output = open(out_fname, 'w')
    P = "\t".join([f"P{x}" for x in range(n)])
    C = "\t".join([f"C{x}" for x in range(n)])
    print(f"Condition\tGenome\tFeatures\tCT\tRep\t{P}\t{C}", file=output)
    for line in results:
        print("\t".join(line), file=output)
    output.close()

main()