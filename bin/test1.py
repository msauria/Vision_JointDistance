#!/usr/bin/env python

import argparse
import gzip
import logging
import multiprocessing
import os
import sys

import numpy

import JointTadsLib

def main():
    parser = generate_parser()
    args = parser.parse_args()
    model = LinReg(args.rna, args.state, args.cre, args.genome, args.verbose)
    model.run(args.output, args.features, args.init_dist, args.promoter_dist,
              args.cre_dist, args.iterations, args.lessone, args.shuffle,
              args.seed)

def generate_parser():
    """Generate an argument parser."""
    description = "%(prog)s -- Predict RNA expression from cCREs and Ideas states"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-r', '--rna', dest="rna", type=str, action='store', required=True,
                        nargs="+", help="RNA expression files")
    parser.add_argument('-s', '--state', dest="state", type=str, action='store', required=True,
                        nargs="+", help="State files")
    parser.add_argument('-c', '--cre', dest="cre", type=str, action='store', required=True,
                        nargs="+", help="CRE files")
    parser.add_argument('-g', '--genome', dest="genome", type=str, action='store', required=True,
                        nargs="+", help="Species names")
    parser.add_argument('-l', '--lessone', dest="lessone", type=str, action='store', default=None,
                        help="Species and cell type to leave out")
    parser.add_argument('-o', '--output', dest="output", type=str, action='store', default='./out',
                        help="Output prefix")
    parser.add_argument('-i', '--iterations', dest="iterations", type=int, action='store', default=100,
                        help="Refinement iterations")
    parser.add_argument('--features', dest="features",
                        choices=['cres', 'promoters', 'both'], default='both',
                        help="Which features should be used for expression prediction")
    parser.add_argument('--initialization-dist', dest="init_dist", type=int, action='store',
                        help="Beta initialization distance cutoff", default=1000)
    parser.add_argument('--promoter-dist', dest="promoter_dist", type=int, action='store',
                        help="If specified, learn betas for promoters up to promoter distance cutoff",
                        default=2500)
    parser.add_argument('--cre-dist', dest="cre_dist", type=int, action='store',
                        help="CRE distance cutoff", default=1000000)
    parser.add_argument('--shuffle', dest="shuffle", default='none',
                        choices=['none', 'cre', 'tss'],
                        help="Shuffle the celltype expression levels amongst genes as a negative control")
    parser.add_argument('--seed', dest="seed", action='store', type=int,
                        help="Random number generator state seed")
    parser.add_argument('-v', '--verbose', dest="verbose", action='store', type=int, default=2,
                        help="Verbosity level")
    return parser


class LinReg(object):
    log_levels = {
        -1: logging.NOTSET,
        0: logging.ERROR,
        1: logging.WARNING,
        2: logging.INFO,
        3: logging.DEBUG,
    }

    def __init__(self, rna, state, cre, genomes, verbose=2):
        self.verbose = verbose
        self.logger = logging.getLogger("Model")
        self.logger.setLevel(self.log_levels[verbose])
        ch = logging.StreamHandler()
        ch.setLevel(self.log_levels[verbose])
        formatter = logging.Formatter(
            fmt='%(asctime)s %(levelname)s %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p'
        )
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        self.logger.propagate = False
        self.rna_fnames = rna
        self.state_fnames = state
        self.cre_fnames = cre
        self.genomes = genomes
        self.genomeN = len(self.genomes)
        assert len(self.cre_fnames) == self.genomeN
        assert len(self.state_fnames) == self.genomeN
        assert len(self.rna_fnames) == self.genomeN
        self.data = []
        self.genome2int = {}
        self.cellN = numpy.zeros(self.genomeN, dtype=numpy.int32)
        self.tss_indices = numpy.zeros(self.genomeN + 1, dtype=numpy.int32)
        self.cell_indices = numpy.zeros(self.genomeN + 1, dtype=numpy.int32)
        for i in range(self.genomeN):
            self.data.append(GenomeData(self.cre_fnames[i], self.state_fnames[i],
                                        self.rna_fnames[i], self.genomes[i], self.verbose))
            self.genome2int[self.genomes[i]] = i
            self.cellN[i] = self.data[i].cellN
            self.cell_indices[i + 1] = self.cell_indices[i] + self.cellN[i]
            self.tss_indices[i + 1] = self.tss_indices[i] + self.data[i].tssN
        self.stateN = self.data[0].stateN

    def __getitem__(self, key):
        """Dictionary-like lookup."""
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            return None

    def __setitem__(self, key, value):
        """Dictionary-like value setting."""
        self.__dict__[key] = value
        return None

    def run(self, out_prefix, features="both", initialization_dist=1000,
            promoter_dist=None, cre_dist=None, iterations=100,
            lessone=None, shuffle='none', seed=None):
        self.out_prefix = out_prefix
        self.use_features = features
        self.initialization_dist = initialization_dist
        self.cre_dist = cre_dist
        self.promoter_dist = promoter_dist
        self.shuffle = shuffle
        self.seed = seed
        if self.seed is None:
            self.seed = numpy.random.randint(0, 100000000)
        self.rng = numpy.random.default_rng(self.seed)
        for i in range(self.genomeN):
            self.data[i].initialize_genome(self.use_features, self.initialization_dist,
                                           self.cre_dist, self.promoter_dist,
                                           self.rng.integers(0, 100000000), self.shuffle)
        self.logger.info("Initialized genomes")
        self.offset = self.data[0].offset
        self.xN = self.data[0].xN

        # Sample to hold out and predict at end
        self.lessone = lessone
        self.lo_mask = []
        self.lint = None
        self.tssN = 0
        for i in range(self.genomeN):
            if self.lessone is None or self.lessone.split(',')[0] != self.genomes[i]:
                self.lo_mask.append(numpy.arange(self.cellN[i]))
            else:
                index = numpy.where(self.data[i].celltypes == self.lessone.split(",")[1])[0][0]
                self.lo_mask.append(numpy.r_[numpy.arange(index),
                                             numpy.arange(index + 1, self.cellN[i])])
                self.lint = (i, index)
                self.cell_indices[(i+1):] -= 1
            self.data[i].set_lo_mask(self.lo_mask[i])
            self.tssN += self.data[i].tssN

        self.iterations = iterations
        self.get_expression()
        if self.use_features != "promoters":
            # Get initial betas
            self.initial_betas = self.get_initial_betas()
            self.logger.info("Got initial betas")
            for i in range(self.genomeN):
                self.data[i].find_tss_cre_pairs(self.initial_betas)
            self.refine_tads()
        self.get_features()
        self.betas = self.get_betas()
        self.write_results()

    def get_features(self, skipped=False):
        features = []
        for i in range(self.genomeN):
            features.append(self.data[i].features[:, self.lo_mask[i], :].reshape(-1, self.xN,
                                                                                 order="c"))
        self.features = numpy.concatenate(features, axis=0)
        if self.lint is not None:
            self.lo_features = self.data[self.lint[0]].features[:, self.lint[1], :]
        if skipped:
            self.skipped_features = []
            temp = []
            for i in range(self.genomeN):
                temp.append(self.data[i].features[:, self.lo_mask[i], :])
            for i in range(self.genomeN):
                self.skipped_features.append([])
                features = []
                for j in range(i):
                    features.append(temp[j].reshape(-1, self.xN, order='c'))
                features.append(None)
                for j in range(i + 1, self.genomeN):
                    features.append(temp[j].reshape(-1, self.xN, order='c'))
                for j in range(self.lo_mask[i].shape[0]):
                    features[i] = temp[i][:,
                                          numpy.r_[numpy.arange(j),
                                                   numpy.arange(j + 1, temp[i].shape[1])],
                                          :].reshape(-1, self.xN, order='c')
                    self.skipped_features[i].append(numpy.concatenate(features, axis=0))

    def get_expression(self):
        expression = []
        for i in range(self.genomeN):
            expression.append(self.data[i].rna['rna'][:, self.lo_mask[i]].reshape(-1, order='c'))
        self.expression = numpy.concatenate(expression)
        if self.lint is not None:
            self.lo_expression = self.data[self.lint[0]].rna['rna'][:, self.lint[1]]
        if self.use_features != 'promoters':
            self.skipped_expression = []
            temp = []
            for i in range(self.genomeN):
                temp.append(self.data[i].rna['rna'][:, self.lo_mask[i]])
            for i in range(self.genomeN):
                self.skipped_expression.append([])
                expression = []
                for j in range(i):
                    expression.append(temp[j].reshape(-1, order='c'))
                expression.append(None)
                for j in range(i + 1, self.genomeN):
                    expression.append(temp[j].reshape(-1, order='c'))
                for j in range(self.lo_mask[i].shape[0]):
                    expression[i] = temp[i][:, numpy.r_[numpy.arange(j),
                                                        numpy.arange(j + 1, temp[i].shape[1])]
                                            ].reshape(-1, order='c')
                    self.skipped_expression[i].append(numpy.concatenate(expression))

    def get_initial_betas(self):
        features = []
        for i in range(self.genomeN):
            features.append(self.data[i].features[:, self.lo_mask[i], self.offset:].reshape(-1, self.stateN,
                                                                                            order="c"))
        features = numpy.concatenate(features, axis=0)
        betas = self.get_betas(features=features)
        if self.use_features == "both":
            betas = numpy.concatenate([numpy.zeros(self.stateN, dtype=numpy.float32), betas])
        return betas

    def get_betas(self, expression=None, features=None):
        if expression is None:
            expression = self.expression
        if features is None:
            features = self.features
        betas = numpy.linalg.lstsq(features, expression, rcond=None)[0]
        return betas

    def refine_tads(self):
        self.get_features()
        self.betas = self.get_betas()
        current_adjR2 = -1
        lo_adjR2 = -1
        best_adjR2 = current_adjR2
        best_tads = []
        for i in range(self.genomeN):
            best_tads.append(None)
        prev_R2 = -1
        for i in range(self.iterations):
            for j in range(self.genomeN):
                self.data[j].refine_tads(self.betas, i)
            self.get_features()
            self.betas = self.get_betas()
            current_adjR2, mse = self.find_adjR2()
            if self.lint is not None:
                lo_adjR2, _ = self.find_adjR2(features=self.lo_features, expression=self.lo_expression)
            else:
                lo_adjR2 = -1
            if current_adjR2 > best_adjR2:
                for j in range(self.genomeN):
                    best_tads[j] = numpy.copy(self.data[j].tads)
                best_adjR2 = current_adjR2
            num_tads, mean_size, changed = self.tad_stats()
            self.logger.info("Iteration: {}   {} TADs (mean size {:.2f}, {} changed)   adj-R2: {:.2f}%%   LO adj-R2: {:.2f}%%  MSE:{:.4f}".format(
                i + 1, num_tads, mean_size, changed, current_adjR2*100, lo_adjR2*100, mse))
            if changed == 0:
                break
            prev_R2 = current_adjR2
        for j in range(self.genomeN):
            self.data[j].tads = numpy.copy(best_tads[j])

    def find_adjR2(self, features=None, expression=None):
        if expression is None:
            expr = numpy.copy(self.expression)
        else:
            expr = numpy.copy(expression)
        pred = self.find_predicted(features)
        mse = numpy.sum((pred - expr) ** 2)
        n = pred.shape[0]
        p = self.xN
        pred -= numpy.mean(pred)
        pred /= numpy.std(pred)
        expr -= numpy.mean(expr)
        expr /= numpy.std(expr)
        R2 = numpy.mean(pred * expr) ** 2
        aR2 = 1 - (1 - R2) * (n - 1) / (n - p - 1)
        return aR2, mse

    def find_predicted(self, features=None):
        if features is None:
            pred = numpy.sum(self.features * self.betas.reshape(1, -1), axis=1)
        else:
            pred = numpy.sum(features * self.betas.reshape(1, -1), axis=1)
        return pred

    def tad_stats(self):
        num_tads = 0
        tad_size = 0
        for i in range(self.genomeN):
            num_tads += self.data[i].tads.shape[0]
            tad_size += numpy.sum(self.data[i].joint_coords[self.data[i].tads[:, 1] - 1] -
                                  self.data[i].joint_coords[self.data[i].tads[:, 0]])
            changed = numpy.sum(self.data[i].changed)
        return num_tads, tad_size / num_tads, changed

    def write_results(self):
        self.write_settings()
        self.write_betas()
        self.write_correlations()
        for i in range(self.genomeN):
            if self.use_features != 'promoters':
                self.data[i].write_tads(self.out_prefix)
            self.data[i].write_expression(self.betas, self.out_prefix)
            if self.shuffle != 'none':
                self.data[i].write_order(self.out_prefix)
            self.data[i].write_pairs(self.betas, self.out_prefix)

    def write_settings(self):
        output = open('{}_settings.txt'.format(self.out_prefix), 'w')
        print("rna file(s) = {}".format(", ".join(self.rna_fnames)), file=output)
        print("state file(s) = {}".format(", ".join(self.state_fnames)), file=output)
        print("cre file(s) = {}".format(", ".join(self.cre_fnames)), file=output)
        print("genome(s) = {}".format(", ".join(self.genomes)), file=output)
        print("features used = {}".format(self.use_features), file=output)
        if self.lessone is not None:
            print("lessone = {}".format(", ".join(self.lessone.split(","))), file=output)
        print("shuffled: {}".format(self.shuffle), file=output)
        if self.use_features != 'cres':
            print("promoter_dist = {}".format(self.promoter_dist), file=output)
        if self.use_features != 'promoters':
            print("initialization_dist = {}".format(self.initialization_dist), file=output)
            print("cre_dist = {}".format(self.cre_dist), file=output)
            print("iterations = {}".format(self.iterations), file=output)
        print("seed = {}".format(self.seed), file=output)
        output.close()

    def write_betas(self):
        output = open("{}_betas.txt".format(self.out_prefix), "w")
        print("Feature\tState\tBeta", file=output)
        if self.use_features != 'cres':
            for i in range(self.stateN):
                print("Promoter\t{}\t{}".format(i, self.betas[i]), file=output)
        if self.use_features != 'promoters':
            for i in range(self.stateN):
                print("CRE\t{}\t{}".format(i, self.betas[i + self.offset]), file=output)
        output.close()

    def write_correlations(self):
        output = open("{}_correlations.txt".format(self.out_prefix), "w")
        print("Gneome\tCelltype\tAdjR2", file=output)
        for i in range(self.genomeN):
            for j in range(self.data[i].cellN):
                adjR2, _ = self.find_adjR2(features=self.data[i].features[:, j, :],
                                        expression=self.data[i].rna['rna'][:, j])
                print("{}\t{}\t{}".format(self.genomes[i], self.data[i].celltypes[j], adjR2),
                                          file=output)
            adjR2, _ = self.find_adjR2(features=self.data[i].features.reshape(-1, self.xN, order="c"),
                                    expression=self.data[i].rna['rna'].reshape(-1, order="c"))
            print("{}\t{}\t{}".format(self.genomes[i], "All", adjR2), file=output)
        features = []
        expression = []
        for i in range(self.genomeN):
            features.append(self.data[i].features.reshape(-1, self.xN, order="c"))
            expression.append(self.data[i].rna['rna'].reshape(-1, order="c"))

        adjR2, _ = self.find_adjR2(features=numpy.concatenate(features, axis=0),
                                expression=numpy.concatenate(expression))
        print("{}\t{}\t{}".format("All", "All", adjR2), file=output)
        output.close()


class GenomeData(object):
    log_levels = {
        -1: logging.NOTSET,
        0: logging.ERROR,
        1: logging.WARNING,
        2: logging.INFO,
        3: logging.DEBUG,
    }

    def __init__(self, cre, state, rna, genome, verbose=2):
        self.verbose = verbose
        self.logger = logging.getLogger(genome)
        self.logger.setLevel(self.log_levels[verbose])
        ch = logging.StreamHandler()
        ch.setLevel(self.log_levels[verbose])
        formatter = logging.Formatter(
            fmt='%(asctime)s %(levelname)s %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p'
        )
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        self.logger.propagate = False

        self.genome = genome
        self.cre_fname = cre
        self.state_fname = state
        self.rna_fname = rna
        self.get_valid_celltypes()
        self.load_rna()
        self.load_CREs()
        self.load_state()
        self.filter_by_chromosomes()

    def __getitem__(self, key):
        """Dictionary-like lookup."""
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            return None

    def __setitem__(self, key, value):
        """Dictionary-like value setting."""
        self.__dict__[key] = value
        return None

    def get_valid_celltypes(self):
        """Find a common set of celltypes between RNA and state files and determine # of reps for each"""
        if self.rna_fname.split('.')[-1] == 'npy':
            rna_header = numpy.load(self.rna_fname).dtype.names[3:]
            rna_celltypes = [x.split('_')[0] for x in rna_header]
        else:
            fs = open(self.rna_fname)
            rna_header = fs.readline().rstrip('\r\n').split()[3:]
            fs.close()
            rna_celltypes = [x.split('_')[0] for x in rna_header]
        if self.state_fname.split('.')[-1] == 'npy':
            state_header = numpy.load(self.state_fname).dtype.names[3:-1]
            state_celltypes = [x.split('_')[0] for x in state_header]
        else:
            fs = open(self.state_fname)
            state_header = fs.readline().rstrip('\r\n').split()[4:-1]
            fs.close()
            state_celltypes = [x.split('_')[0] for x in state_header]
        celltypes = list(set(rna_celltypes).intersection(set(state_celltypes)))
        celltypes.sort()
        celltypes = numpy.array(celltypes)
        self.celltypes = celltypes
        self.cellN = self.celltypes.shape[0]
        self.logger.info("Found {} RNA-state pairings for celltypes {}".format(
                         self.genome, ', '.join(list(self.celltypes))))

    def load_CREs(self):
        if self.verbose >= 2:
            print("\r{}\rLoading {} cCRE data".format(' ' * 80, self.genome), end='', file=sys.stderr)
        if self.cre_fname.split('.')[-1] != 'npy':
            fs = gzip.open(self.cre_fname, 'rb')
            _ = fs.readline().rstrip(b'\r\n').split()
            data = []
            chromlen = 0
            for line in fs:
                line = line.rstrip(b'\r\n').split()
                chrom, start, end = line[:3]
                if chrom not in self.chr2int:
                    continue
                data.append((chrom.decode('utf8'), int(start), int(end)))
                chromlen = max(chromlen, len(chrom))
            fs.close()
            data = numpy.array(data, dtype=numpy.dtype([('chr', 'U%i' % chromlen),
                                                        ('start', numpy.int32),
                                                        ('end', numpy.int32)]))
        else:
            data = numpy.load(self.cre_fname)
        data.sort(order=["chr", "start"])
        valid = numpy.zeros(data.shape[0], dtype=bool)
        for chrom in self.chroms:
            valid[numpy.where(data['chr'] == chrom)] = True
        valid = numpy.where(valid)[0]
        self.cre = data[valid]
        self.cre_indices = numpy.r_[0, numpy.where(self.cre['chr'][1:] != self.cre['chr'][:-1])[0] + 1,
                                    self.cre.shape[0]]
        self.creN = self.cre.shape[0]
        self.chromN = self.chroms.shape[0]
        if self.verbose >= 2:
            print("\r{}\r".format(' ' * 80), end='', file=sys.stderr)
        self.logger.info('Loaded {} {} CREs'.format(self.cre_indices[-1], self.genome))

    def load_rna(self):
        if self.verbose >= 2:
            print("\r{}\rLoading {} RNA data".format(' ' * 80, self.genome), end='', file=sys.stderr)
        if self.rna_fname.split('.')[-1] == 'npy':
            temp = numpy.load(self.rna_fname)
            header = temp.dtype.names
        else:
            fs = open(self.rna_fname)
            header = fs.readline().rstrip('\r\n').split()
            strand2bool = {'+': False, '-': True}
            temp = []
            for line in fs:
                line = line.rstrip('\r\n').split()
                chrom, tss, strand = line[:3]
                if chrom not in self.chr2int:
                    continue
                temp.append(tuple([chrom, int(tss), gene, strand2bool[strand]] + line[3:]))
            fs.close()
            dtype = [('chr', self.chroms.dtype), ('TSS', numpy.int32), ('strand', bool)]
            for name in header[3:]:
                dtype.append((name, numpy.float32))
            temp = numpy.array(temp, dtype=dtype)
        temp.sort(order=["chr", "TSS"])
        self.chroms = numpy.unique(temp['chr'])
        self.chr2int = {}
        for i, chrom in enumerate(self.chroms):
            self.chr2int[chrom] = i
        self.rna_cindices = numpy.zeros(self.cellN + 1, dtype=numpy.int32)
        for name in header[3:]:
            where = numpy.where(self.celltypes == name.split('_')[0])[0]
            if where.shape[0] > 0:
                self.rna_cindices[where[0] + 1] += 1
        self.rna_cindices = numpy.cumsum(self.rna_cindices)
        self.rna_reps = self.rna_cindices[1:] - self.rna_cindices[:-1]
        self.rnaRepN = self.rna_cindices[-1]
        data = numpy.empty(temp.shape[0], dtype=numpy.dtype([
            ('chr', temp['chr'].dtype), ('TSS', numpy.int32), ('strand', bool),
            ('rna', numpy.float32, (self.rnaRepN,))]))
        for name in header[:3]:
            data[name] = temp[name]
        pos = numpy.copy(self.rna_cindices[:-1])
        for name in header[3:]:
            where = numpy.where(self.celltypes == name.split('_')[0])[0]
            if where.shape[0] == 0:
                continue
            data['rna'][:, pos[where[0]]] = temp[name]
            pos[where[0]] += 1
        rna = numpy.zeros((data['rna'].shape[0], self.cellN), dtype=numpy.float32)
        for i in range(self.cellN):
            rna[:, i] = numpy.mean(data['rna'][:, self.rna_cindices[i]:self.rna_cindices[i + 1]], axis=1)
        self.rna = numpy.empty(data.shape[0], dtype=numpy.dtype([
                ('chr', data['chr'].dtype), ('TSS', numpy.int32), ('strand', bool),
                ('start', numpy.int32), ('end', numpy.int32),
                ('rna', numpy.float32, (self.cellN,))]))
        self.rna['chr'] = data['chr']
        self.rna['TSS'] = data['TSS']
        self.rna['strand'] = data['strand']
        self.rna['rna'] = numpy.copy(rna)
        self.rna_indices = numpy.zeros(self.chroms.shape[0] + 1, dtype=numpy.int32)
        for i, chrom in enumerate(self.chroms):
            self.rna_indices[i + 1] = (self.rna_indices[i]
                                       + numpy.where(self.rna['chr'] == chrom)[0].shape[0])
        self.tssN = self.rna.shape[0]
        if self.verbose >= 2:
            print("\r%s\r" % (' ' * 80), end='', file=sys.stderr)
        self.logger.info('Loaded {} expression profiles across {} {} celltypes ({} replicates)'.format(
                         self.tssN, self.cellN, self.genome, self.rnaRepN))

    def load_state(self):
        if self.verbose >= 2:
            print("\r{}\rLoading {} state data".format(' ' * 80, self.genome), end='', file=sys.stderr)
        if self.state_fname.split('.')[-1] == 'npy':
            temp = numpy.load(self.state_fname)
            header = temp.dtype.names[3:]
        else:
            fs = open(self.state_fname)
            header = fs.readline().rstrip('\n\r').split()[4:-1]
            temp = []
            for line in fs:
                line = line.rstrip('\r\n').split()
                binID, chrom, start, end = line[:4]
                if chrom not in self.chr2int:
                    continue
                temp.append(tuple([chrom, int(start), int(end)] + line[4:-1]))
            fs.close()
            dtype = [('chr', self.chroms.dtype), ('start', numpy.int32), ('end', numpy.int32)]
            for name in header[4:]:
                dtype.append((name, numpy.int32))
            temp = numpy.array(temp, dtype=dtype)
            temp.sort(order=["chr", "start"])
        valid = numpy.zeros(temp.shape[0], dtype=bool)
        for chrom in self.chroms:
            valid[numpy.where(temp['chr'] == chrom)] = True
        valid = numpy.where(valid)[0]
        self.state_cindices = numpy.zeros(self.cellN + 1, dtype=numpy.int32)
        for name in header:
            where = numpy.where(self.celltypes == name.split('_')[0])[0]
            if where.shape[0] > 0:
                self.state_cindices[where[0] + 1] += 1
        self.state_cindices = numpy.cumsum(self.state_cindices)
        self.state_reps = self.state_cindices[1:] - self.state_cindices[:-1]
        self.stateRepN = self.state_cindices[-1]
        data = numpy.empty(valid.shape[0], dtype=numpy.dtype([
            ('chr', temp['chr'].dtype), ('start', numpy.int32), ('end', numpy.int32),
            ('state', numpy.int32, (self.stateRepN,))]))
        for name in temp.dtype.names[:3]:
            data[name] = temp[name][valid]
        pos = numpy.copy(self.state_cindices[:-1])
        for name in header:
            where = numpy.where(self.celltypes == name.split('_')[0])[0]
            if where.shape[0] == 0:
                continue
            data['state'][:, pos[where[0]]] = temp[name][valid]
            pos[where[0]] += 1
        self.stateN = numpy.amax(data['state']) + 1
        self.state_indices = numpy.zeros(self.chroms.shape[0] + 1, dtype=numpy.int32)
        for i, chrom in enumerate(self.chroms):
            where = numpy.where(data['chr'] == chrom)[0]
            self.state_indices[i + 1] = self.state_indices[i] + where.shape[0]
        self.state = data
        if self.verbose >= 2:
            print("\r{}\r".format(' ' * 80), end='', file=sys.stderr)
        self.logger.info('Loaded {} non-zero state profiles ({} states) across {} {} celltypes ({} replicates)'.format(
                         self.state.shape[0], self.stateN, self.cellN, self.genome, self.stateRepN))

    def filter_by_chromosomes(self):
        r_chroms = set(numpy.unique(self.rna['chr']))
        c_chroms = set(numpy.unique(self.cre['chr']))
        s_chroms = set(numpy.unique(self.state['chr']))
        chroms = numpy.array(list(r_chroms.intersection(c_chroms.intersection(s_chroms))),
                             dtype='U')
        chroms.sort()
        self.chroms = chroms
        for name in ['cre', 'rna', 'state']:
            data = self[name]
            valid = numpy.zeros(data.shape[0], bool)
            indices = numpy.zeros(chroms.shape[0] + 1, dtype=numpy.int32)
            for i, chrom in enumerate(chroms):
                where = numpy.where(data['chr'] == chrom)[0]
                valid[where] = True
                indices[i + 1] = indices[i] + where.shape[0]
            self[name] = data[numpy.where(valid)]
            self["{}_indices".format(name)] = indices
        self.creN = self.cre.shape[0]
        self.tssN = self.rna.shape[0]
        self.chromN = self.chroms.shape[0]
        self.chr2int = {}
        for i, chrom in enumerate(chroms):
            self.chr2int[chrom] = i
        self.logger.info("After filtering, using {} chromosomes {}".format(
            self.genome, ", ".join(list(self.chroms))))
        self.logger.info("After filtering, {} CREs: {}  States: {}  TSSs: {}".format(
            self.genome, self.creN, self.state.shape[0], self.tssN))

    def initialize_genome(self, features, initialization_dist, cre_dist, promoter_dist,
                          seed, shuffle):
        self.rng = numpy.random.default_rng(seed)
        self.shuffle = shuffle
        self.use_features = features
        self.initialization_dist = initialization_dist
        self.cre_dist = cre_dist
        self.promoter_dist = promoter_dist
        if self.shuffle == 'tss':
            self.order = numpy.arange(self.tssN)
            self.rng.shuffle(self.order)
            self.rna['rna'] = self.rna['rna'][self.order, :]
        self.find_initial_features()

    def assign_promoter_states(self, pdist):
        """Find the proportion of states in each promoter window"""
        if self.verbose >= 2:
            print("\r{}\rAssign states to {} promoters".format(' ' * 80, self.genome),
                  end='', file=sys.stderr)
        self.rna['start'] = self.rna['TSS'] - pdist
        self.rna['end'] = self.rna['TSS'] + pdist
        # Find ranges of states for each CRE
        Pranges = numpy.zeros((self.rna.shape[0], 2), dtype=numpy.int32)
        for i in range(self.rna_indices.shape[0] - 1):
            s = self.rna_indices[i]
            e = self.rna_indices[i + 1]
            if e - s == 0:
                continue
            s1 = self.state_indices[i]
            e1 = self.state_indices[i + 1]
            if e1 - s1 == 0:
                continue
            starts = numpy.searchsorted(self.state['end'][s1:e1],
                                        self.rna['start'][s:e],
                                        side='right') + s1
            stops = numpy.searchsorted(self.state['start'][s1:e1],
                                       self.rna['end'][s:e],
                                       side='left') + s1
            Pranges[s:e, 0] = starts
            Pranges[s:e, 1] = stops - 1
        self.Pranges = Pranges
        # Even though there may be multiple reps for a celltype, we only find the average state proportion across reps
        Pstates2 = numpy.zeros((self.tssN, self.stateRepN, self.stateN), dtype=numpy.int32)
        r = numpy.arange(self.stateRepN)
        for i in  range(self.tssN):
            start, stop = Pranges[i, :]
            TSS_st = self.rna['start'][i]
            TSS_ed = self.rna['end'][i]
            dist = TSS_ed - TSS_st
            if stop < start:
                Pstates2[i, :, 0] = dist
                continue
            overlap = stop - start
            size = max(0, (min(TSS_ed, self.state['end'][start])
                    - max(TSS_st, self.state['start'][start])))
            Pstates2[i, r, self.state['state'][start, :]] = size
            if overlap >= 1:
                size = max(0, (min(TSS_ed, self.state['end'][stop])
                        - max(TSS_st, self.state['start'][stop])))
                Pstates2[i, r, self.state['state'][stop, :]] += size
            if overlap >= 2:
                for j in range(start + 1, stop):
                    size = self.state['end'][j] - self.state['start'][j]
                    Pstates2[i, r, self.state['state'][j, :]] += size
            Pstates2[i, :, 0] = dist - numpy.sum(Pstates2[i, :, 1:], axis=1)
        Pstates = numpy.zeros((self.tssN, self.cellN, self.stateN), dtype=numpy.float32)
        for i in range(self.cellN):
            s = self.state_cindices[i]
            e = self.state_cindices[i + 1]
            Pstates[:, i, :] = numpy.sum(Pstates2[:, s:e, :], axis=1) / (e - s)
        self.Pstates = Pstates / (self.rna['end'] - self.rna['start']).reshape(-1, 1, 1)
        if self.shuffle == 'cre':
            self.order2 = numpy.arange(self.tssN)
            self.rng.shuffle(self.order2)
            self.Pstates = self.Pstates[self.order2, :, :]
        if self.verbose >= 2:
            print("\r{}\r".format(' ' * 80), end='', file=sys.stderr)

    def assign_CRE_states(self):
        """Find the proportion of states in each cCRE"""
        if self.verbose >= 2:
            print("\r{}\rAssign states to {} CREs".format(' ' * 80, self.genome),
                  end='', file=sys.stderr)
        # Find ranges of states for each CRE
        Cranges = numpy.zeros((self.cre.shape[0], 2), dtype=numpy.int32)
        for i in range(self.cre_indices.shape[0] - 1):
            s = self.cre_indices[i]
            e = self.cre_indices[i + 1]
            if e - s == 0:
                continue
            s1 = self.state_indices[i]
            e1 = self.state_indices[i + 1]
            if e1 - s1 == 0:
                continue
            starts = numpy.searchsorted(self.state['end'][s1:e1],
                                        self.cre['start'][s:e], side='right') + s1
            stops = numpy.searchsorted(self.state['start'][s1:e1],
                                       self.cre['end'][s:e], side='left') + s1
            Cranges[s:e, 0] = starts
            Cranges[s:e, 1] = stops
        self.Cranges = Cranges
        # Even though there may be multiple reps for a celltype, we only find the average state proportion across reps
        Cstates = numpy.zeros((self.cre.shape[0], self.cellN, self.stateN), dtype=numpy.float32)
        Cstates2 = numpy.zeros((self.cre.shape[0], self.stateRepN, self.stateN), dtype=numpy.float32)
        r = numpy.arange(self.stateRepN)
        for i in range(Cranges.shape[0]):
            start, stop = Cranges[i, :]
            Cstart, Cstop = self.Cranges[i, 0], self.Cranges[i, 1] - 1
            if Cstop < Cstart:
                Cstates2[i, :, 0] = self.cre['end'][i] - self.cre['start'][i]
                continue
            overlap = Cstop - Cstart
            size = (min(self.cre['end'][i], self.state['end'][Cstart])
                    - max(self.cre['start'][i], self.state['start'][Cstart]))
            Cstates2[i, r, self.state['state'][Cstart, :]] = size
            if overlap >= 1:
                size = (min(self.cre['end'][i], self.state['end'][Cstop])
                        - max(self.cre['start'][i], self.state['start'][Cstop]))
                Cstates2[i, r, self.state['state'][Cstop, :]] += size
            if overlap >= 2:
                for j in range(Cstart + 1, Cstop):
                    size = self.state['end'][j] - self.state['start'][j]
                    Cstates2[i, r, self.state['state'][j, :]] += size
            Cstates2[i, :, 0] = (self.cre['end'][i] - self.cre['start'][i]
                                 - numpy.sum(Cstates2[i, :, 1:], axis=1))
        for i in range(self.cellN):
            s = self.state_cindices[i]
            e = self.state_cindices[i + 1]
            Cstates[:, i, :] = numpy.sum(Cstates2[:, s:e, :], axis=1) / (e - s)
        self.Cstates = Cstates / (self.cre['end'] - self.cre['start']).reshape(-1, 1, 1)
        if self.shuffle == 'cre':
            self.order = numpy.arange(self.cre.shape[0])
            self.rng.shuffle(self.order)
            self.Cstates = self.Cstates[self.order, :, :]
        if self.verbose >= 2:
            print("\r{}\r".format(' ' * 80), end='', file=sys.stderr)

    def find_initial_features(self):
        self.offset = 0
        if self.use_features != "both":
            self.xN = self.stateN
        else:
            self.xN = 2 * self.stateN
        self.features = numpy.zeros((self.tssN, self.cellN, self.xN), dtype=numpy.float32)
        if self.use_features != "promoters":
            self.assign_CRE_states()
            if self.use_features == "both":
                self.offset = self.stateN
            tss_ranges = numpy.zeros((self.tssN, 2), dtype=numpy.int32)
            tss_indices = numpy.zeros(self.tssN + 1, dtype=numpy.int32)
            for i in range(self.chromN):
                tstart = self.rna_indices[i]
                tend = self.rna_indices[i + 1]
                s = self.cre_indices[i]
                e = self.cre_indices[i + 1]
                cre = self.cre[s:e]
                tss = self.rna[tstart:tend]
                tss_ranges[tstart:tend, 0] = numpy.searchsorted(
                    cre['end'], tss['TSS'] - self.initialization_dist, side='right') + s
                tss_ranges[tstart:tend, 1] = numpy.searchsorted(
                    cre['start'], tss['TSS'] + self.initialization_dist, side='left') + s
            tss_indices[1:] = numpy.cumsum(tss_ranges[:, -1] - tss_ranges[:, 0])
            pairs = numpy.zeros((tss_indices[-1], 2), dtype=numpy.int32)
            for i in range(self.tssN):
                s, e = tss_indices[i:(i + 2)]
                if e == s or numpy.amin(self.rna['rna'][i, :]) == numpy.amax(self.rna['rna'][i, :]):
                    continue
                pairs[s:e, 0] = i
                pairs[s:e, 1] = numpy.arange(tss_ranges[i, 0], tss_ranges[i, -1])
            pair_indices = numpy.r_[0, numpy.cumsum(numpy.bincount(pairs[:, 0],
                                                                   minlength=self.tssN))]
            for i in range(self.tssN):
                s, e = pair_indices[i:(i+2)]
                if e == s:
                    continue
                self.features[i, :, self.offset:] = numpy.sum(self.Cstates[pairs[s:e, 1], :, :], axis=0)

        if self.use_features != "cres":
            self.assign_promoter_states(self.promoter_dist)
            self.features[:, :, :self.stateN] = self.Pstates

    def set_lo_mask(self, lo_mask):
        self.lo_mask = lo_mask
        self.lo_N = self.lo_mask.shape[0]

    def find_tss_cre_pairs(self, betas):
        if self.verbose >= 2:
            print("\r{}\rFinding {} TSS-cCRE pairs".format(' ' * 80, self.genome),
                  end='', file=sys.stderr)
        joint_EP = numpy.zeros((self.tssN + self.creN, 4), dtype=numpy.int32)
        joint_coords = numpy.zeros(self.tssN + self.creN, dtype=numpy.int32)
        joint_indices = numpy.zeros(self.chromN + 1, dtype=numpy.int32)
        for i in range(self.chromN):
            tstart = self.rna_indices[i]
            tend = self.rna_indices[i + 1]
            s = self.cre_indices[i]
            e = self.cre_indices[i + 1]
            cre = self.cre[s:e]
            tss = self.rna[tstart:tend]
            cre_mid = (cre['start'] + cre['end']) // 2
            coords = numpy.r_[cre_mid, tss['TSS']]
            joint_indices[i + 1] = joint_indices[i] + coords.shape[0]
            js, je = joint_indices[i:(i+2)]
            joint_EP[js:je, 0] = numpy.r_[numpy.arange(s, e),
                                              numpy.arange(tstart, tend)]
            joint_EP[js:je, 1] = numpy.r_[numpy.zeros(e - s, dtype=numpy.int32),
                                              numpy.ones(tend - tstart, dtype=numpy.int32)]
            order = numpy.argsort(coords)
            coords = coords[order]
            joint_EP[js:je, :] = joint_EP[order + js, :]
            joint = joint_EP[js:je, :]
            TSSs = numpy.where(joint[:, 1] == 1)[0]
            up_indices = numpy.searchsorted(coords, coords[TSSs] - self.cre_dist, side='right')
            for j, k in enumerate(TSSs):
                l = k
                while l > up_indices[j]:
                    if joint[l - 1, 1] == 1:
                        break
                    l -= 1
                joint[k, 2] = l + js
            down_indices = numpy.searchsorted(coords, coords + self.cre_dist, side='right')
            joint[:, 3] = down_indices + js
            joint_EP[js:je, :] = joint
            joint_coords[js:je] = coords
        self.joint_EP = joint_EP
        self.joint_coords = joint_coords
        self.joint_indices = joint_indices
        self.jointN = self.joint_indices[-1]
        self.changed = 0
        self.tads = numpy.zeros((0, 2), dtype=numpy.int32)
        print("\r{}\r".format(' ' * 80), end='', file=sys.stderr)

    def find_features(self):
        for s, e in self.tads:
            TSSs = numpy.where(self.joint_EP[s:e, 1] == 1)[0] + s
            CREs = numpy.where(self.joint_EP[s:e, 1] == 0)[0] + s
            if CREs.shape[0] > 0:
                features = numpy.sum(self.Cstates[self.joint_EP[CREs, 0], :, :], axis=0)
                self.features[self.joint_EP[TSSs, 0], :, self.offset:] = features
            else:
                self.features[self.joint_EP[TSSs, 0], :, self.offset:] = 0

    def find_MSE(self, betas, expression=None, features=None):
        if expression is None:
            expression = self.rna['rna'][:, self.lo_mask]
        if features is None:
            features = self.features[:, self.lo_mask, :]
        predicted = numpy.sum(features * betas.reshape(1, -1, self.xN), axis=2)
        MSE = numpy.sum((predicted - expression) ** 2) / expression.size
        return MSE

    def refine_tads(self, betas, iteration):
        prev_tads = set()
        for s, e in self.tads:
            prev_tads.add((s, e))
        MSE = self.find_new_tads(betas, iteration)
        new_tads = set()
        for s, e in self.tads:
            new_tads.add((s, e))
        self.changed = (len(prev_tads) + len(new_tads) -
                        2 * len(new_tads.intersection(prev_tads)))
        return MSE

    def find_new_tads(self, betas, iteration):
        new_tads = []
        for c in range(self.chromN):
            if self.verbose > 2:
                print("\r{}\rFinding TADs for {}".format(" "*80, self.chroms[c]),
                      end="", file=sys.stderr)
            cstart, cend = self.cre_indices[c:(c+2)]
            cStateBetas = numpy.sum(self.Cstates[cstart:cend, self.lo_mask, :] *
                                    betas[self.offset:].reshape(1, 1, -1), axis=2).astype(
                                    numpy.float32)
            tstart, tend = self.rna_indices[c:(c+2)]
            if self.use_features != 'cres':
                pStateBetas = numpy.sum(self.features[tstart:tend, self.lo_mask, :self.offset] *
                                        betas[:self.offset].reshape(1, 1, -1), axis=2).astype(
                                        numpy.float32)
            else:
                pStateBetas = numpy.zeros((tend - tstart, self.lo_N),
                                          dtype=numpy.float32)
            RNA = numpy.copy(self.rna['rna'][tstart:tend, self.lo_mask])
            jstart, jend = self.joint_indices[c:(c+2)]
            joint = numpy.copy(self.joint_EP[jstart:jend, :])
            coords = self.joint_coords[jstart:jend]
            TSSs = numpy.where(joint[:, 1] == 1)[0].astype(numpy.int32)
            cre = numpy.where(joint[:, 1] == 0)[0]
            joint[TSSs, 0] -= tstart
            joint[cre, 0] -= cstart
            tads = numpy.zeros((RNA.shape[0], 2), dtype=numpy.int32)
            scores = numpy.empty(joint.shape[0] + 1, dtype=numpy.float32)
            paths = numpy.empty(joint.shape[0] + 1, dtype=numpy.int32)
            cStateSum = numpy.zeros(self.lo_N, dtype=numpy.float32)
            TSS_list = numpy.zeros(RNA.shape[0], dtype=numpy.int32)
            tadN = JointTadsLib.find_new_tads1(joint,
                                              coords,
                                              RNA,
                                              cStateBetas,
                                              pStateBetas,
                                              scores,
                                              paths,
                                              cStateSum,
                                              TSS_list,
                                              tads,
                                              self.cre_dist)
            tads[:tadN, :] += jstart
            for i in range(tadN)[::-1]:
                new_tads.append((tads[i, 0], tads[i, 1]))
        new_tads.sort()
        self.tads = numpy.array(new_tads, dtype=numpy.int32)
        self.find_features()
        total_MSE = numpy.sum((numpy.sum(self.features[:, self.lo_mask, :] *
                                         betas.reshape(1, 1, -1), axis=2) -
                               self.rna['rna'][:, self.lo_mask]) ** 2)
        if self.verbose > 2:
            print("\r{}\r".format(" "*80), end="", file=sys.stderr)
        return total_MSE

    def find_new_tads_old(self, betas):
        new_tads = []
        for c in range(self.chromN):
            jstart, jend = self.joint_indices[c:(c+2)]
            joint = self.joint_EP[jstart:jend, :]
            cstart, cend = self.cre_indices[c:(c+2)]
            CRE = (self.Cstates[cstart:cend, self.lo_mask, :] *
                   betas[self.offset:].reshape(1, 1, -1))
            tstart, tend = self.rna_indices[c:(c+2)]
            rna = numpy.copy(self.rna['rna'][tstart:tend, self.lo_mask])
            if self.use_features == 'both':
                rna -= numpy.sum(self.features[tstart:tend, self.lo_mask, :self.offset] *
                                 betas[:self.offset].reshape(1, 1, -1), axis=2)
            prev_best = 0
            prev_path = 0
            scores = numpy.full(joint.shape[0] + 1, numpy.nan, dtype=numpy.float32)
            path = numpy.zeros(joint.shape[0] + 1, dtype=numpy.int32)
            path[0] = -1
            TSSs = numpy.where(joint[:, 1] == 1)[0]
            for i in TSSs:
                for j in range(joint[i, 2] - jstart, i + 1)[::-1]:
                    if numpy.isnan(scores[j]):
                        k = j - 1
                        while k >= 0:
                            if not numpy.isnan(scores[k]):
                                prev_best = scores[k]
                                prev_path = path[k]
                                break
                            k -= 1
                    for k in range(i + 1, joint[j, 3]):
                        t = numpy.where(joint[j:k, 1] == 1)[0]
                        c = numpy.where(joint[j:k, 1] == 0)[0]
                        expr = rna[joint[t, 0] - tstart, :]
                        if c.shape[0] > 0:
                            pred = numpy.sum(numpy.sum(CRE[joint[c, 0] - cstart, :, :] *
                                                       betas[self.offset:].reshape(1, 1, -1),
                                                       axis=2), axis=0).reshape(1, -1)
                            new_score = numpy.sum((pred - expr) ** 2)
                        else:
                            new_score = numpy.sum(expr ** 2)
                        if numpy.isnan(scores[j]):
                            new_score += prev_best
                            new_path = prev_path
                        else:
                            new_score += scores[j]
                            new_path = j
                        if new_score < scores[k]:
                            scores[k] = new_score
                            path[k] = new_path
            i = TSSs[-1] + 1
            end = i
            best_score = scores[i]
            while i < joint.shape[0] and not numpy.isnan(scores[i + 1]):
                if scores[i + 1] < best_score:
                    best_score = scores[i + 1]
                    end = [i + 1]
                i += 1
            start = path[end]
            while start <= 0:
                if numpy.sum(joint[start:end, 1]) > 0:
                    new_tads.append((start + jstart, end + jstart))
                end = start
                start = path[end]
        new_tads.sort()
        self.tads = numpy.array(new_tads, dtype=numpy.int32)
        self.find_features()
        total_MSE = numpy.sum((numpy.sum(self.features[:, self.lo_mask, :] *
                                         betas.reshape(1, 1, -1), axis=2) -
                               self.rna['rna'][:, self.lo_mask]) ** 2)
        return total_MSE

    def write_tads(self, out_prefix):
        output = open("{}_{}_tads.bed".format(out_prefix, self.genome), "w")
        chrints = numpy.searchsorted(self.joint_indices[1:], self.tads[:, 0],
                                     side='right')
        for i in range(self.tads.shape[0]):
            chrint = chrints[i]
            chrom = self.chroms[chrint]
            s, e = self.tads[i, :]
            if s > self.joint_indices[chrint]:
                start = (self.joint_coords[s - 1] + self.joint_coords[s]) // 2
            else:
                start = max(0, self.joint_coords[s] - 1000)
            if e < self.joint_indices[chrint + 1] - 1:
                end = (self.joint_coords[e - 1] + self.joint_coords[e]) // 2
            else:
                end = self.joint_coords[e - 1] + 1000
            print("{}\t{}\t{}".format(chrom, start, end), file=output)
        output.close()

    def write_expression(self, betas, out_prefix):
        output = open("{}_{}_expression.txt".format(out_prefix, self.genome), "w")
        expression = numpy.sum(self.features * betas.reshape(1, 1, -1), axis=2)
        print("TSS-Chr\tTSS-Coord\t{}".format("\t".join([f"eRP-{x}" for x in self.celltypes])), file=output)
        for j in range(self.tssN):
            gene = self.rna[j]
            print("{}\t{}\t{}".format(
                gene['chr'], gene['TSS'], "\t".join(["{:.4f}".format(x) for x in expression[j, :]])),
                file=output)
        output.close()

    def write_order(self, out_prefix):
        if self.shuffle =='tss':
            output = open("{}_{}_gene_order.txt".format(out_prefix, self.genome), "w")
        elif self.use_features != 'promoters':
            output = open("{}_{}_cre_order.txt".format(out_prefix, self.genome), "w")
        for i in self.order:
            print(i, file=output)
        output.close()
        if self.shuffle == 'cre' and self.use_features != 'cres':
            output = open("{}_{}_promoter_order.txt".format(out_prefix, self.genome), "w")
            for i in self.order2:
                print(i, file=output)
            output.close()

    def write_pairs(self, betas, prefix):
        output = open("{}_{}_pairs.txt".format(prefix, self.genome), 'w')
        temp = ['chr', 'TSS', 'cre-start', 'cre-end']
        for ct in self.celltypes:
            temp.append('eRP-{}'.format(ct))
        print("{}".format("\t".join(temp)), file=output)
        for h in numpy.arange(self.tads.shape[0]):
            s, e = self.tads[h, :]
            temp = self.joint_EP[s:e, :]
            TSSs = temp[numpy.where(temp[:, 1] == 1)[0], 0]
            CREs = temp[numpy.where(temp[:, 1] == 0)[0], 0]
            if CREs.shape[0] > 0:
                eRPs = numpy.sum(self.Cstates[CREs, :, :] *
                                 betas[self.offset:].reshape(1, 1, -1), axis=2)
            for i in TSSs:
                TSS = self.rna[i]
                if self.use_features != "cres":
                    promoter = numpy.sum(self.features[i, :, :self.offset] *
                                         betas[:self.offset].reshape(1, -1), axis=1)
                    print("{}\t{}\t{}\t{}\t{}".format(
                        TSS['chr'], TSS['TSS'], -1, -1,
                        "\t".join([str(x) for x in promoter])), file=output)
                for j, c in enumerate(CREs):
                    CRE = self.cre[c]
                    print("{}\t{}\t{}\t{}\t{}".format(
                        TSS['chr'], TSS['TSS'], CRE['start'], CRE['end'],
                        "\t".join([str(x) for x in eRPs[j, :]])), file=output)
        output.close()


if __name__ == "__main__":
    main()
