#!/usr/bin/env python

import argparse
import gzip
import logging
import multiprocessing
import os
import sys

import numpy

import JointDistanceLib

def main():
    parser = generate_parser()
    args = parser.parse_args()
    model = LinReg(args.rna, args.state, args.cre, args.genome, args.verbose)
    model.run(args.output, args.promoter_dist, args.cre_dist, args.unidir,
              args.beta_iter, args.tad_iter, args.lessone, args.shuffle,
              args.npy, args.rand, args.seed)

def generate_parser():
    """Generate an argument parser."""
    description = "%(prog)s -- Predict RNA expression from cCREs and Ideas states"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-r', '--rna', dest="rna", type=str, action='store', required=True,
                        nargs="+", help="RNA expression files")
    parser.add_argument('-s', '--state', dest="state", type=str, action='store', required=True,
                        nargs="+", help="State files")
    parser.add_argument('-c', '--cre', dest="cre", type=str, action='store',
                        nargs="+", help="CRE files")
    parser.add_argument('-g', '--genome', dest="genome", type=str, action='store', required=True,
                        nargs="+", help="Species names")
    parser.add_argument('-l', '--lessone', dest="lessone", type=str, action='store', default=None,
                        help="Species and cell type to leave out")
    parser.add_argument('-o', '--output', dest="output", type=str, action='store', default='./out',
                        help="Output prefix")
    parser.add_argument('-i', '--beta-iter', dest="beta_iter", type=int, action='store', default=20,
                        help="Parameter refinement iterations")
    parser.add_argument('-t', '--tad-iter', dest="tad_iter", type=int, action='store', default=100,
                        help="TAD refinement iterations")
    parser.add_argument('--promoter-dist', dest="promoter_dist", type=int, action='store',
                        help="If specified, learn betas for promoters up to promoter distance cutoff",
                        default=2500)
    parser.add_argument('--cre-dist', dest="cre_dist", type=int, action='store',
                        help="If specified, learn betas for CREs up to CRE distance cutoff",
                        default=250000)
    parser.add_argument('--unidirectional', dest="unidir", action='store_true',
                        help="Restrict promoter to upstream side only")
    parser.add_argument('--npy', dest="npy", action='store_true',
                        help="Store large outputs as npy files")
    parser.add_argument('--rand-init', dest="rand", action='store_true',
                        help="Randomly initialize betas")
    parser.add_argument('--shuffle', dest="shuffle", action='store_true',
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
    learning_rate = 0.0000002

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
        assert len(self.state_fnames) == self.genomeN
        assert len(self.rna_fnames) == self.genomeN
        if self.cre_fnames is not None:
            assert len(self.cre_fnames) == self.genomeN
            self.use_cres = True
        else:
            self.cre_fnames = [None for x in range(self.genomeN)]
            self.use_cres = False
        self.data = []
        self.genome2int = {}
        self.cellN = numpy.zeros(self.genomeN, dtype=numpy.int32)
        self.tss_indices = numpy.zeros(self.genomeN + 1, dtype=numpy.int32)
        self.cell_indices = numpy.zeros(self.genomeN + 1, dtype=numpy.int32)
        for i in range(self.genomeN):
            self.data.append(GenomeData(self.state_fnames[i], self.rna_fnames[i],
                                        self.cre_fnames[i], self.genomes[i],
                                        self.verbose))
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

    def run(self, out_prefix, promoter_dist=None, cre_dist=None, unidir=False,
            beta_iter=20, tad_iter=100, lessone=None, shuffle=False, npy=False,
            rand=False, seed=None):
        self.out_prefix = os.path.abspath(out_prefix)
        self.promoter_dist = promoter_dist
        if self.promoter_dist == 0:
            self.promoter_dist = None
        self.cre_dist = cre_dist
        if self.cre_dist == 0 or not self.use_cres:
            self.cre_dist = None
            self.use_cres = False
        self.use_promoters = self.promoter_dist is not None
        self.featureN = self.use_promoters + self.use_cres
        if self.featureN == 0:
            raise RuntimeError("Either promoter_dist or cre_dist must be specified")
        self.unidir = unidir
        self.shuffle = shuffle
        self.seed = seed
        if self.seed is None:
            self.seed = numpy.random.randint(0, 100000000)
        self.rng = numpy.random.default_rng(self.seed)
        self.randinit = rand
        self.npy = npy
        for i in range(self.genomeN):
            self.data[i].initialize_genome(self.promoter_dist, self.cre_dist,
                                           self.unidir,
                                           self.rng.integers(0, 100000000),
                                           self.npy,
                                           self.shuffle)
        self.logger.info("Initialized genomes")
        self.stateN = self.data[0].stateN

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

        self.beta_iter = beta_iter
        self.tad_iter = tad_iter
        self.get_expression()
        self.initialize_parameters()
        self.refine_model()
        self.write_results()

    def get_expression(self):
        expression = []
        expr_norm = []
        for i in range(self.genomeN):
            expr = self.data[i].rna['rna'][:, self.lo_mask[i]]
            expression.append(numpy.copy(expr).reshape(-1, order='c'))
            expr -= numpy.mean(expr, axis=0, keepdims=True)
            expr /= numpy.std(expr, axis=0, keepdims=True)
            expr_norm.append(expr.reshape(-1, order='c'))
        self.expression = numpy.concatenate(expression).astype(numpy.float64)
        self.expr_norm = numpy.concatenate(expr_norm).astype(numpy.float64)
        if self.lint is not None:
            self.lo_expression = self.data[self.lint[0]].rna['rna'][:, self.lint[1]]
            if self.use_cres:
                self.lo_contactP = self.data[self.lint[0]].contactP
            else:
                self.lo_contactP = None
            self.lo_expr_norm = numpy.copy(self.lo_expression)
            self.lo_expr_norm -= numpy.mean(self.lo_expression)
            std = numpy.std(self.lo_expression)
            if not numpy.isnan(std) and std > 0:
                self.lo_expr_norm /= std

    def get_predicted(self):
        predicted = []
        pred_norm = []
        for i in range(self.genomeN):
            p = self.data[i].find_predicted(self.betas)[:, self.data[i].lo_mask]
            predicted.append(numpy.copy(p).reshape(-1, order='c'))
            p -= numpy.mean(p, axis=0, keepdims=True)
            std =  numpy.std(p, axis=0)
            p[:, numpy.where(std > 0)[0]] /= std[numpy.where(std > 0)[0]].reshape(1, -1)
            pred_norm.append(p.reshape(-1, order='c'))
            if self.lint is not None and i == self.lint[0]:
                self.lo_predicted = p[:, self.lint[i]]
                self.lo_pred_norm = numpy.copy(self.lo_predicted)
                self.lo_pred_norm -= numpy.mean(self.lo_pred_norm)
                std = numpy.std(self.lo_pred_norm)
                if std > 0:
                    self.lo_pred_norm /= std
        self.predicted = numpy.concatenate(predicted, axis=0)
        self.pred_norm = numpy.concatenate(pred_norm, axis=0)

    def initialize_parameters(self):
        self.betas = numpy.zeros((1, self.featureN, self.stateN),
                                 dtype=numpy.float64)
        self.alpha = numpy.array([-1], dtype=numpy.float64)
        if self.use_promoters:
            features = []
            for i in range(self.genomeN):
                features.append(self.data[i].pfeatures[
                    :, self.data[i].lo_mask, :].reshape(-1, self.stateN, order='c'))
            features = numpy.concatenate(features, axis=0)
            betas = numpy.linalg.lstsq(
                features, self.expression, rcond=None)[0]
            self.betas[0, 0, :] = betas
        for i in range(self.genomeN):
            self.data[i].update_contactP(self.alpha)

    def refine_model(self):
        prefix = self.out_prefix
        if self.use_cres and self.randinit:
            self.betas[0, 0, :] = self.rng.random(self.stateN) * 5
            self.betas[0, -1, :] = self.rng.random(self.stateN) * 5
            for j in range(self.genomeN):
                self.data[j].update_TADs(self.betas)
        self.get_predicted()
        adjR2, lo_adjR2, MSE = self.find_adjR2()
        if self.use_cres:
            tadN, tadS, tadC = self.tad_stats()
            self.logger.info(f"Initial  " + \
                  f"adj-R2: {adjR2*100:.2f}%  " + \
                  f"LO adj-R2: {lo_adjR2*100:.2f}%  " + \
                  f"MSE:{MSE:,.2f}  " + \
                  f"{tadN} TADs ({tadS}Kb, {tadC} changed)")
            bgradient = numpy.zeros((self.featureN, self.stateN),
                                    dtype=numpy.float64)
            agradient = numpy.zeros(1, dtype=numpy.float64)
            for h in range(self.tad_iter):
                for i in range(self.beta_iter):
                    bgradient.fill(0)
                    agradient.fill(0)
                    for j in range(self.genomeN):
                        self.data[j].update_parameters(self.betas,
                                                       agradient, bgradient)
                    if h == 0:
                        #self.alpha = numpy.minimum(-0.1, numpy.maximum(
                        #    self.alpha + self.learning_rate *
                        #    agradient * 0.1, -2))
                        self.betas[0, :, :] = numpy.minimum(20, numpy.maximum(
                            self.betas[0, :, :] + self.learning_rate *
                            bgradient * 0.1, -20))
                    else:
                        #self.alpha = numpy.minimum(-0.1, numpy.maximum(
                        #    self.alpha - self.learning_rate * agradient, -2))
                        self.betas[0, :, :] = numpy.minimum(20, numpy.maximum(
                            self.betas[0, :, :] + self.learning_rate *
                            bgradient, -20))
                    for j in range(self.genomeN):
                        self.data[j].update_contactP(self.alpha)
                    self.get_predicted()
                    adjR2, lo_adjR2, MSE = self.find_adjR2()
                    print(f"\r{' '*80}\rIteration: {h+1} ({i+1})  " + \
                          f"adj-R2: {adjR2*100:.2f}%  " + \
                          f"LO adj-R2: {lo_adjR2*100:.2f}%  " + \
                          f"MSE:{MSE:,.2f}", end='', file=sys.stderr)
                print(f"\r{' '*80}\r", end='', file=sys.stderr)
                self.logger.info(f"Iteration: {h+0.5}  " + \
                                 f"adj-R2: {adjR2*100:.2f}%  " + \
                                 f"LO adj-R2: {lo_adjR2*100:.2f}%  " + \
                                 f"MSE:{MSE:,.2f}")
                for j in range(self.genomeN):
                    self.data[j].update_TADs(self.betas)
                self.get_predicted()
                adjR2, lo_adjR2, MSE = self.find_adjR2()
                tadN, tadS, tadC = self.tad_stats()
                self.logger.info(f"Iteration: {h+1}    " + \
                                 f"adj-R2: {adjR2*100:.2f}%  " + \
                                 f"LO adj-R2: {lo_adjR2*100:.2f}%  " + \
                                 f"MSE:{MSE:,.2f}  " + \
                                 f"{tadN} TADs ({tadS}Kb, {tadC} changed)")
        self.logger.info(f"Final  " + \
              f"adj-R2: {adjR2*100:.2f}%  " + \
              f"LO adj-R2: {lo_adjR2*100:.2f}%  " + \
              f"MSE:{MSE:,.2f}")

    def find_adjR2(self):
        MSE = numpy.sum((self.predicted - self.expression) ** 2)
        n = self.predicted.shape[0]
        p = self.betas.size + 1
        R2 = numpy.mean(self.pred_norm * self.expr_norm) ** 2
        aR2 = 1 - (1 - R2) * (n - 1) / (n - p - 1)
        if self.lint is not None:
            n = self.lo_pred_norm.shape[0]
            p = self.betas.size + 1
            R2 = numpy.mean(self.lo_pred_norm * self.lo_expr_norm) ** 2
            lo_aR2 = 1 - (1 - R2) * (n - 1) / (n - p - 1)
        else:
            lo_aR2 = 0
        return aR2, lo_aR2, MSE

    def tad_stats(self):
        num_tads = 0
        tad_size = 0
        for i in range(self.genomeN):
            num_tads += self.data[i].tads.shape[0]
            tad_size += numpy.sum(
                self.data[i].EP[self.data[i].tads[:, 1] - 1, 2] -
                self.data[i].EP[self.data[i].tads[:, 0], 2])
            changed = numpy.sum(self.data[i].changed)
        return num_tads, int(tad_size / num_tads / 1000), changed

    def write_results(self):
        if not os.path.isdir(os.path.split(self.out_prefix)[0]):
            os.makedirs(os.path.split(self.out_prefix)[0])
        self.write_settings()
        self.write_betas()
        self.write_correlations()
        for i in range(self.genomeN):
            self.data[i].write_predicted(self.betas, self.out_prefix)
            if self.use_cres:
                self.data[i].write_pairs(self.betas, self.out_prefix)
                self.data[i].write_tads(self.out_prefix)
            if self.shuffle:
                self.data[i].write_order(self.out_prefix)

    def write_settings(self):
        output = open(f'{self.out_prefix}_settings.txt', 'w')
        print("rna file(s) = {}".format(", ".join(self.rna_fnames)), file=output)
        print("state file(s) = {}".format(", ".join(self.state_fnames)), file=output)
        print("genome(s) = {}".format(", ".join(self.genomes)), file=output)
        if self.lessone is not None:
            print("lessone = {}".format(", ".join(self.lessone.split(","))), file=output)
        print("shuffled: {}".format(self.shuffle), file=output)
        print("promoter_dist = {}".format(self.promoter_dist), file=output)
        print("cre_dist = {}".format(self.cre_dist), file=output)
        print("unidirectional_promoter = {}".format(self.unidir), file=output)
        print("beta iterations = {}".format(self.beta_iter), file=output)
        print("TAD iterations = {}".format(self.tad_iter), file=output)
        print("seed = {}".format(self.seed), file=output)
        output.close()

    def write_betas(self):
        output = open(f"{self.out_prefix}_betas.txt", "w")
        print("Feature\tState\tBeta", file=output)
        features = []
        if self.use_promoters:
            features.append('Promoter')
        if self.use_cres:
            features.append('CRE')
        for i in range(self.featureN):
            for j in range(self.stateN):
                print(f"{features[i]}\t{j}\t{self.betas[0, i, j]}", file=output)
        print(f"alpha\t{self.alpha[0]}", file=output)
        output.close()

    def write_correlations(self):
        output = open("{}_correlations.txt".format(self.out_prefix), "w")
        print("Genome\tCelltype\tAdjR2", file=output)
        self.get_predicted()
        predicted = numpy.copy(self.predicted)
        expression = numpy.copy(self.expression)
        norm_exp = numpy.copy(self.expr_norm)
        norm_pred = numpy.copy(self.pred_norm)
        pos = 0
        for i in range(self.genomeN):
            self.predicted = predicted[
                pos:(pos + self.data[i].lo_N * self.data[i].tssN)]
            self.expression = expression[
                pos:(pos + self.data[i].lo_N * self.data[i].tssN)]
            self.expr_norm = norm_exp[
                pos:(pos + self.data[i].lo_N * self.data[i].tssN)]
            self.pred_norm = norm_pred[
                pos:(pos + self.data[i].lo_N * self.data[i].tssN)]
            if self.lint is not None and self.lint[0] == i:
                self.predicted = numpy.concatenate(
                    (self.predicted, self.lo_predicted), axis=0)
                self.expression = numpy.concatenate(
                    (self.expression, self.lo_expression), axis=0)
                self.expr_norm = numpy.concatenate(
                    (self.expr_norm, self.lo_expr_norm), axis=0)
                self.pred_norm = numpy.concatenate(
                    (self.pred_norm, self.lo_pred_norm), axis=0)
            adjR2, _, _ = self.find_adjR2()
            print(f"{self.genomes[i]}\tAll\t{adjR2}", file=output)
            for j in range(self.data[i].cellN):
                if self.lint is not None and self.lint[0] == i and self.lint[1] == j:
                    self.predicted = self.lo_predicted
                    self.expression = self.lo_expression
                    self.expr_norm = self.lo_expr_norm
                    self.pred_norm = self.lo_pred_norm
                else:
                    self.predicted = predicted[pos:(pos + self.data[i].tssN)]
                    self.expression = expression[pos:(pos + self.data[i].tssN)]
                    self.expr_norm = norm_exp[pos:(pos + self.data[i].tssN)]
                    self.pred_norm = norm_pred[pos:(pos + self.data[i].tssN)]
                    pos += self.data[i].tssN
                adjR2, _, _ = self.find_adjR2()
                print(f"{self.genomes[i]}\t{self.data[i].celltypes[j]}\t{adjR2}",
                      file=output)
        self.predicted = predicted
        self.expression = expression
        self.expr_norm = norm_exp
        self.pred_norm = norm_pred
        adjR2, _, _ = self.find_adjR2()
        print(f"All\tAll\t{adjR2}", file=output)
        output.close()


class GenomeData(object):
    log_levels = {
        -1: logging.NOTSET,
        0: logging.ERROR,
        1: logging.WARNING,
        2: logging.INFO,
        3: logging.DEBUG,
    }
    mindist = 100

    def __init__(self, state, rna, cre, genome, verbose=2):
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
        self.state_fname = state
        self.rna_fname = rna
        self.cre_fname = cre
        self.use_cres = self.cre_fname is not None
        self.get_valid_celltypes()
        self.load_rna()
        self.load_state()
        if self.use_cres:
            self.load_CREs()
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

    def load_rna(self):
        if self.verbose >= 2:
            print(f"\r{' '*80}\rLoading {self.genome} RNA data",
                  end='', file=sys.stderr)
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
                dtype.append((name, numpy.float64))
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
            ('rna', numpy.float64, (self.rnaRepN,))]))
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
                ('rna', numpy.float64, (self.cellN,))]))
        self.rna['chr'] = data['chr']
        self.rna['TSS'] = data['TSS']
        self.rna['strand'] = data['strand']
        self.rna['rna'] = numpy.copy(rna)

        std = numpy.mean(numpy.std(self.rna['rna'], axis=0))
        self.rna['rna'] /= numpy.std(self.rna['rna'], axis=0, keepdims=True)
        self.rna['rna'] *= std

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
            print(f"\r{' '*80}\rLoading {self.genome} state data",
                  end='', file=sys.stderr)
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

    def load_CREs(self):
        if self.verbose >= 2:
            print(f"\r{' '*80}\rLoading {self.genome} cCRE data",
                  end='', file=sys.stderr)
        data = numpy.load(self.cre_fname)
        data.sort(order=["chr", "start"])
        valid = numpy.zeros(data.shape[0], dtype=bool)
        for chrom in self.chroms:
            valid[numpy.where(data['chr'] == chrom)] = True
        valid = numpy.where(valid)[0]
        self.cre = numpy.zeros(valid.shape[0],
                               dtype=numpy.dtype([('chr', data['chr'].dtype),
                                                  ('start', numpy.int32),
                                                  ('end', numpy.int32),
                                                  ('mid', numpy.int32),
                                                  ('state', numpy.float64,
                                                   (self.cellN, self.stateN))]))
        for i in ['chr', 'start', 'end']:
            self.cre[i] = data[i][valid]
        self.cre['mid'] = (self.cre['start'] + self.cre['end']) // 2
        self.cre_indices = numpy.r_[0, numpy.where(self.cre['chr'][1:] !=
                                                   self.cre['chr'][:-1])[0] + 1,
                                    self.cre.shape[0]]
        self.creN = self.cre.shape[0]
        self.chromN = self.chroms.shape[0]
        if self.verbose >= 2:
            print("\r{}\r".format(' ' * 80), end='', file=sys.stderr)
        self.logger.info('Loaded {} {} CREs'.format(self.cre_indices[-1], self.genome))

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

    def initialize_genome(self, promoter_dist, cre_dist,
                          unidir, seed, npy, shuffle):
        self.rng = numpy.random.default_rng(seed)
        self.shuffle = shuffle
        self.promoter_dist = promoter_dist
        self.use_promoters = self.promoter_dist is not None
        self.cre_dist = cre_dist
        self.use_cres = self.use_cres and self.cre_dist is not None
        self.unidir = unidir
        self.npy = npy
        if self.shuffle:
            self.order = numpy.arange(self.tssN)
            self.rng.shuffle(self.order)
            self.rna['rna'] = self.rna['rna'][self.order, :]
        if self.use_promoters:
            self.assign_promoter_states()
        if self.use_cres:
            self.assign_CRE_states()
            self.assign_EP_pairs()

    def assign_promoter_states(self):
        """Find the proportion of states in each promoter window"""
        if self.verbose >= 2:
            print(f"\r{' '*80}\rAssign states to {self.genome} promoters",
                  end='', file=sys.stderr)
        if self.unidir:
            where = numpy.where(self.rna['strand'] == False)[0]
            self.rna['start'][where] = self.rna['TSS'][where] - self.promoter_dist
            self.rna['end'][where] = self.rna['TSS'][where]
            where = numpy.where(self.rna['strand'] == True)[0]
            self.rna['start'][where] = self.rna['TSS'][where]
            self.rna['end'][where] = self.rna['TSS'][where] - self.promoter_dist
        else:
            self.rna['start'] = self.rna['TSS'] - self.promoter_dist
            self.rna['end'] = self.rna['TSS'] + self.promoter_dist
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
        # Even though there may be multiple reps for a celltype,
        # we only find the average state proportion across reps
        Pstates2 = numpy.zeros((self.tssN, self.stateRepN, self.stateN),
                               dtype=numpy.int32)
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
        Pstates = numpy.zeros((self.tssN, self.cellN, self.stateN),
                              dtype=numpy.float64)
        for i in range(self.cellN):
            s = self.state_cindices[i]
            e = self.state_cindices[i + 1]
            Pstates[:, i, :] = numpy.sum(Pstates2[:, s:e, :], axis=1) / (e - s)
        self.pfeatures = Pstates / (self.rna['end'] -
                                    self.rna['start']).reshape(-1, 1, 1)
        if self.verbose >= 2:
            print(f"\r{' '*80}\r", end='', file=sys.stderr)

    def assign_CRE_states(self):
        """Find the proportion of states in each cCRE"""
        if self.verbose >= 2:
            print(f"\r{' '*80}\rAssign states to {self.genome} CREs",
                  end='', file=sys.stderr)
        # Find ranges of states for each CRE
        Cranges = numpy.zeros((self.cre.shape[0], 2), dtype=numpy.int32)
        for i in range(self.cre_indices.shape[0] - 1):
            s, e = self.cre_indices[i:(i+2)]
            if e - s == 0:
                continue
            s1, e1 = self.state_indices[i:(i+2)]
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
        self.cre['state'] = Cstates / (self.cre['end'] -
                                       self.cre['start']).reshape(-1, 1, 1)
        if self.shuffle == 'cre':
            self.order = numpy.arange(self.cre.shape[0])
            self.rng.shuffle(self.order)
            self.Cstates = self.Cstates[self.order, :, :]
        if self.verbose >= 2:
            print(f"\r{' '*80}\r", end='', file=sys.stderr)

    def assign_EP_pairs(self):
        """Find the proportion of states in cres in cre window, scaled by distance"""
        if self.verbose >= 2:
            print(f"\r{' '*80}\rAssign {self.genome} EP pairs",
                  end='', file=sys.stderr)
        self.cre_ranges = numpy.zeros((self.tssN, 2), dtype=numpy.int32)
        self.EP_indices = numpy.zeros(self.chromN + 1, dtype=numpy.int32)
        EP = []
        tads = []
        self.max_tss = 0
        for i in range(self.rna_indices.shape[0] - 1):
            ts, te = self.rna_indices[i:(i + 2)]
            cs, ce = self.cre_indices[i:(i + 2)]
            es = self.EP_indices[i]
            starts = numpy.searchsorted(self.cre['mid'][cs:ce],
                                        self.rna['TSS'][ts:te] - self.cre_dist) + cs
            ends = numpy.searchsorted(self.cre['mid'][cs:ce],
                                      self.rna['TSS'][ts:te] + self.cre_dist) + cs
            self.cre_ranges[ts:te, 0] = starts
            self.cre_ranges[ts:te, 1] = ends
            EP.append(numpy.zeros((te - ts + ce - cs, 5), dtype=numpy.int32))
            EP[-1][:(te - ts), 0] = numpy.arange(ts, te)
            EP[-1][:(te - ts), 1] = 1
            EP[-1][:(te - ts), 2] = self.rna['TSS'][ts:te]
            EP[-1][(te - ts):, 0] = numpy.arange(cs, ce)
            EP[-1][(te - ts):, 2] = self.cre['mid'][cs:ce]
            EP[-1] = EP[-1][numpy.argsort(EP[-1][:, 2]), :]
            tss = numpy.where(EP[-1][:, 1] == 1)[0]
            upstream = numpy.searchsorted(EP[-1][:, 2],
                                          EP[-1][tss, 2] - self.cre_dist)
            downstream = numpy.searchsorted(EP[-1][:, 2],
                                            EP[-1][tss, 2] + self.cre_dist,
                                            side='right')
            EP[-1][tss, 4] = downstream
            for j, k in enumerate(tss):
                where = numpy.where(EP[-1][upstream[j]:k, 1] == 1)[0]
                if where.shape[0] > 0:
                    upstream[j] = where[-1] + 1 + upstream[j]
                self.max_tss = max(self.max_tss, numpy.sum(EP[-1][k:downstream[j], 1]))
            EP[-1][tss, 3] = upstream
            self.EP_indices[i + 1] = es + EP[-1].shape[0]
            if tss[0] > 0:
                valid_cres = numpy.where(EP[-1][tss[0], 2] - EP[-1][:tss[0], 2] <=
                                         self.cre_dist)[0]
                if valid_cres.shape[0] > 0:
                    start = valid_cres[0] + es
                else:
                    start = tss[0] + es
            else:
                start = tss[0] + es
            for j in range(tss.shape[0] - 1):
                if tss[j] + 1 == tss[j + 1]:
                    tads.append([start, tss[j + 1] + es])
                    start = tss[j + 1] + es
                    continue
                dists0 = numpy.abs(
                    EP[-1][tss[j], 2] - EP[-1][(tss[j] + 1):tss[j + 1], 2])
                dists1 = numpy.abs(
                    EP[-1][tss[j + 1], 2] - EP[-1][(tss[j] + 1):tss[j + 1], 2])
                where = numpy.where((dists0 < dists1) & (dists0 <= self.cre_dist))[0]
                tads.append([start, tss[j] + where.shape[0] + 1 + es])
                where = numpy.where((dists0 > dists1) & (dists1 <= self.cre_dist))[0]
                start = tss[j + 1] - where.shape[0] + es
            if tss[-1] + 1 < self.EP_indices[i + 1]:
                dists = EP[-1][(tss[-1] + 1):, 2] - EP[-1][tss[-1], 2]
                where = numpy.where(dists <= self.cre_dist)[0]
                tads.append([start, tss[-1] + 1 + where.shape[0] + es])
            else:
                tads.append([start, tss[-1] + 1 + es])
        self.tads = numpy.array(tads, dtype=numpy.int32)
        self.tadN = self.tads.shape[0]

        self.changed = 0
        self.EP = numpy.concatenate(EP, axis=0)
        self.max_cre = numpy.amax(self.cre_ranges[:, 1] - self.cre_ranges[:, 0])
        self.EP_distances = numpy.zeros((self.tssN, self.max_cre, 2),
                                        dtype=numpy.float64)
        self.contactP = numpy.zeros((self.tssN, self.max_cre),
                                    dtype=numpy.float64)
        for i in range(self.rna_indices.shape[0] - 1):
            ts, te = self.rna_indices[i:(i + 2)]
            for j in range(te - ts):
                s, e = self.cre_ranges[ts + j, :]
                self.EP_distances[ts + j, :(e - s), 0] = numpy.maximum(
                    numpy.abs(self.cre['mid'][s:e] - self.rna['TSS'][ts + j]),
                    self.mindist) / 1000 
        self.EP_nonzero = numpy.where(self.EP_distances[:, :, 0] > 0)
        self.EP_distances[self.EP_nonzero[0], self.EP_nonzero[1], 1] = numpy.log(
            self.EP_distances[self.EP_nonzero[0], self.EP_nonzero[1], 0])
        if self.verbose >= 2:
            print(f"\r{' '*80}\r", end='', file=sys.stderr)

    def set_lo_mask(self, lo_mask):
        self.lo_mask = lo_mask
        self.lo_N = self.lo_mask.shape[0]

    def update_contactP(self, alpha):
        if self.use_cres:
            self.contactP[self.EP_nonzero] = self.EP_distances[
                self.EP_nonzero[0], self.EP_nonzero[1], 0] ** alpha[0]

    def find_predicted(self, betas):
        predicted = numpy.zeros((self.tssN, self.cellN), dtype=numpy.float64)
        if not self.use_cres:
            predicted = numpy.sum(self.pfeatures * betas, axis=2)
            return predicted
        if self.use_promoters:
            pfeatures = self.pfeatures
        else:
            pfeatures = None
        cBetas = numpy.zeros((self.creN, self.cellN), dtype=numpy.float64)
        tss_list = numpy.zeros(self.max_tss, dtype=numpy.int32)
        cre_list = numpy.zeros(self.max_cre, dtype=numpy.int32)
        JointDistanceLib.predicted(
            self.tads,
            self.EP,
            self.rna['rna'],
            pfeatures,
            self.cre['state'],
            self.cre_ranges,
            self.contactP,
            betas,
            tss_list,
            cre_list,
            cBetas,
            predicted)
        return predicted

    def update_parameters(self, betas, agradient, bgradient):
        predicted = self.find_predicted(betas)
        dC_dP = numpy.zeros(self.cellN, dtype=numpy.float64)
        tss_list = numpy.zeros(self.max_tss, dtype=numpy.int32)
        cre_list = numpy.zeros(self.max_cre, dtype=numpy.int32)
        if self.use_promoters:
            pfeatures = self.pfeatures[:, self.lo_mask, :]
        else:
            pfeatures = None
        tss = numpy.where(self.EP[:, 1] == 1)[0].astype(numpy.int32)
        JointDistanceLib.gradient(
            self.tads,
            self.EP,
            self.rna['rna'][:, self.lo_mask],
            pfeatures,
            self.cre['state'][:, self.lo_mask, :],
            self.cre_ranges,
            self.contactP,
            self.EP_distances,
            tss,
            betas,
            tss_list,
            cre_list,
            agradient,
            bgradient,
            predicted,
            dC_dP)

    def update_TADs(self, betas):
        prev_tads = set()
        for s, e in self.tads:
            prev_tads.add((s, e))
        tss_list = numpy.zeros(self.max_tss, dtype=numpy.int32)
        cre_list = numpy.zeros(self.max_cre, dtype=numpy.int32)
        TSSs = numpy.where(self.EP[:, 1])[0]
        self.tads = numpy.zeros((TSSs.shape[0], 2), dtype=numpy.int32)
        self.tads[:, 0] = TSSs
        self.tads[:, 1] = TSSs + 1
        predicted = self.find_predicted(betas)[:, self.lo_mask]
        self.tads = []
        tad_pred = numpy.zeros((self.max_tss, self.cellN), dtype=numpy.float64)
        mse = 0
        for i in range(self.chromN):
            print(f"\r{' '*80}\rFinding {self.genome} {self.chroms[i]} TADs",
                  end='', file=sys.stderr)
            es, ee = self.EP_indices[i:(i + 2)]
            rs, re = self.rna_indices[i:(i + 2)]
            cs, ce = self.cre_indices[i:(i + 2)]
            EP = numpy.copy(self.EP[es:ee, :])
            cre_ranges = self.cre_ranges[rs:re, :] - cs
            rna = self.rna['rna'][rs:re, self.lo_mask]
            contactP = self.contactP[rs:re, :]
            cBetas = numpy.zeros((ce - cs, self.cellN), dtype=numpy.float64)
            inTad_cres = numpy.zeros((re - rs, self.cellN, self.max_cre), dtype=numpy.float64)
            cfeatures = self.cre['state'][cs:ce, self.lo_mask, :]
            tss = numpy.where(EP[:, 1] == 1)[0].astype(numpy.int32)
            cre = numpy.where(EP[:, 1] == 0)[0]
            EP[tss, 0] -= rs
            EP[cre, 0] -= cs
            paths = numpy.zeros(EP.shape[0] + 1, dtype=numpy.int32)
            scores = numpy.zeros(EP.shape[0] + 1, dtype=numpy.float64)
            tads = numpy.zeros((rna.shape[0], 2), dtype=numpy.int32)
            tadN = JointDistanceLib.find_tads(
                EP,
                rna,
                cfeatures,
                cre_ranges,
                contactP,
                tss,
                betas,
                predicted[rs:re, :],
                tad_pred,
                cBetas,
                inTad_cres,
                tss_list,
                cre_list,
                paths,
                scores,
                tads)
            #print(f"  {i} {numpy.amax(scores[(tss[-1] + 1):])} {scores[tads[0, 1]]}", file=sys.stderr)
            self.tads.append(tads[:tadN, :] + es)
            mse += scores[tads[0, 1]]
        self.tads = numpy.concatenate(self.tads, axis=0)
        self.tads = self.tads[numpy.argsort(self.tads[:, 0]), :]
        self.tadN = self.tads.shape[0]
        new_tads = set()
        for s, e in self.tads:
            new_tads.add((s, e))
        self.changed = (len(prev_tads) + len(new_tads) -
                        2 * len(new_tads.intersection(prev_tads)))
        print(f"\r{' '*80}\r", end='', file=sys.stderr)
        #print(f"{mse}", file=sys.stderr)

    def write_tads(self, out_prefix):
        output = open("{}_{}_tads.bed".format(out_prefix, self.genome), "w")
        for i in range(self.tads.shape[0]):
            ts, te = self.tads[i, :]
            if self.EP[ts, 1] == 1:
                start = self.rna['TSS'][self.EP[ts, 0]] - 1
                chrom = self.rna['chr'][self.EP[ts, 0]]
            else:
                start = self.cre['start'][self.EP[ts, 0]] - 1
                chrom = self.cre['chr'][self.EP[ts, 0]]
            if self.EP[te, 1] == 1:
                end = self.rna['TSS'][self.EP[te, 0]] + 1
            else:
                end = self.cre['end'][self.EP[te, 0]] + 1
            print(f"{chrom}\t{start}\t{end}", file=output)
        output.close()

    def write_predicted(self, betas, out_prefix):
        predicted = self.find_predicted(betas)
        dtypes = [("TSS-Chr", self.rna['chr'].dtype),
                  ("TSS-Coord", numpy.int32)]
        dtypes += [(f'{x}', numpy.float64) for x in self.celltypes]
        results = numpy.zeros(self.tssN, dtype=numpy.dtype(dtypes))
        results["TSS-Chr"] = self.rna['chr']
        results["TSS-Coord"] = self.rna['TSS']
        for i, ct in enumerate(self.celltypes):
            results[ct] = predicted[:, i]
        if self.npy:
            numpy.save(f"{out_prefix}_{self.genome}_expression.npy", results)
        else:
            output = open(f"{out_prefix}_{self.genome}_expression.txt", "w")
            print("\t".join(results.dtype.names), file=output)
            for i in range(self.tssN):
                gene = results[i]
                print(f"{gene['TSS-Chr']}\t{gene['TSS-Coord']}\t",
                      end="", file=output)
                temp = '\t'.join([f"{gene[x]:.4f}" for x in self.celltypes])
                print(f"{temp}", file=output)
            output.close()

    def write_pairs(self, betas, out_prefix):
        pairN = 0
        for i in range(self.tadN):
            ts, te = self.tads[i, :]
            tss = numpy.where(self.EP[ts:te, 1] == 1)[0].shape[0]
            cre = numpy.where(self.EP[ts:te, 1] == 0)[0].shape[0]
            pairN += tss * cre
        if self.use_promoters:
            pairN += self.tssN
        indices = numpy.zeros((pairN, 2), dtype=numpy.int32)
        erps = numpy.zeros((pairN, self.cellN), dtype=numpy.float64)
        cBetas = numpy.zeros((self.creN, self.cellN), dtype=numpy.float64)
        tss_list = numpy.zeros(self.max_tss, dtype=numpy.int32)
        cre_list = numpy.zeros(self.max_cre, dtype=numpy.int32)
        if self.use_promoters:
            pfeatures = self.pfeatures
            pBetas = numpy.zeros((self.tssN, self.cellN), dtype=numpy.float64)
        else:
            pfeatures = None
            pBetas = None
        JointDistanceLib.find_erps(
            self.EP,
            self.cre['state'],
            pfeatures,
            self.cre_ranges,
            self.contactP,
            self.tads,
            betas,
            cBetas,
            pBetas,
            tss_list,
            cre_list,
            erps,
            indices)
        dtypes = [('chr', self.rna['chr'].dtype),
                  ('TSS', numpy.int32),
                  ('cre-start', numpy.int32),
                  ('cre-end', numpy.int32)]
        dtypes += [(f'{x}', numpy.float64) for x in self.celltypes]
        pairs = numpy.zeros(pairN, dtype=numpy.dtype(dtypes))
        pairs['chr'] = self.rna['chr'][indices[:, 0]]
        pairs['TSS'] = self.rna['TSS'][indices[:, 0]]
        where = numpy.where(indices[:, 1] >= 0)[0]
        pairs['cre-start'][where] = self.cre['start'][indices[where, 1]]
        pairs['cre-end'][where] = self.cre['end'][indices[where, 1]]
        where = numpy.where(indices[:, 1] == -1)[0]
        pairs['cre-start'][where] = -1
        pairs['cre-end'][where] = -1
        for i, j in enumerate(self.celltypes):
            pairs[j] = erps[:, i]
        if self.npy:
            print(f"{out_prefix}_{self.genome}_pairs.npy")
            numpy.save(f"{out_prefix}_{self.genome}_pairs.npy", pairs)
        else:
            output = open(f"{out_prefix}_{self.genome}_pairs.txt", 'w')
            print("\t".join([x for x in pairs.dtype.names]), file=output)
            for i in range(pairN):
                temp = "\t".join([f"{pairs[ct][i]:.4f}" for ct in self.celltypes])
                print(f"{pairs['chr'][i]}\t{pairs['TSS'][i]}\t" + \
                      f"{pairs['cre-start'][i]}\t{pairs['cre-start'][i]}\t" + \
                      f"{temp}", file=output)
            output.close()

    def write_order(self, out_prefix):
        output = open(f"{out_prefix}_{self.genome}_gene_order.txt", "w")
        for i in self.order:
            print(i, file=output)
        output.close()


if __name__ == "__main__":
    main()
