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
	model = LinReg(args.rna, args.state, args.cre, args.genome, args.verbose, not args.nearest_gene) #addition of , not args.nearest_gene
	model.run(args.output, args.init_dist, args.promoter_dist,
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
	parser.add_argument('--nearest-gene', dest="nearest_gene", action='store_true', # addition of store_true parser argument if wanting to select based on nearest gene rather than selecting and refining
						help="match cCREs to their nearest gene rather than performing the selection process") # addition of help for parser argument
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

	def __init__(self, rna, state, cre, genomes, verbose=2, select_refine=True): #addition of  select_refine = True
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
		self.select_refine = select_refine  #addition of a self.select_refine variable that is true if selection/refinement should be run
											#(e.g., --nearest-gene argument not passed to parser)
											# and false if selection should be based off of the nearest gene
											#(e.g.,  --nearest-gene argument is passed to parser)

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

	def run(self, out_prefix, initialization_dist=1000,
			promoter_dist=None, cre_dist=None, iterations=100,
			lessone=None, shuffle='none', seed=None):
		self.out_prefix = out_prefix
		self.initialization_dist = initialization_dist
		self.cre_dist = cre_dist
		self.promoter_dist = promoter_dist
		self.shuffle = shuffle
		self.seed = seed
		if self.seed is None:
			self.seed = numpy.random.randint(0, 100000000)
		self.rng = numpy.random.default_rng(self.seed)
		for i in range(self.genomeN):
			self.data[i].initialize_genome(self.initialization_dist,
										   self.cre_dist, self.promoter_dist,
										   self.rng.integers(0, 100000000), self.shuffle)
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

		self.iterations = iterations
		self.get_expression()
		self.get_features()
		self.get_betas()
		self.logger.info("Got initial betas")

		for i in range(self.genomeN):
			self.data[i].find_tss_cre_pairs()
		if self.select_refine:
			self.refine_tads()
		else:
			for i in range(self.genomeN):
				self.data[i].find_nearest_gene_pairs()
				self.data[i].find_features_ng()
		self.get_features()
		self.get_betas()
		self.write_results()

	def get_features(self):
		features = []
		for i in range(self.genomeN):
			features.append(self.data[i].pfeatures[
				:, self.lo_mask[i], :].reshape(-1, self.stateN, order="c"))
		self.pfeatures = numpy.concatenate(features, axis=0)
		features = []
		for i in range(self.genomeN):
			features.append(self.data[i].cfeatures[
				:, self.lo_mask[i], :].reshape(-1, self.stateN, order="c"))
		self.cfeatures = numpy.concatenate(features, axis=0)
		if self.lint is not None:
			self.lo_pfeatures = self.data[self.lint[0]].pfeatures[:, self.lint[1], :]
			self.lo_cfeatures = self.data[self.lint[0]].cfeatures[:, self.lint[1], :]

	def get_expression(self):
		expression = []
		for i in range(self.genomeN):
			expression.append(self.data[i].rna['rna'][
				:, self.lo_mask[i]].reshape(-1, order='c'))
		self.expression = numpy.concatenate(expression)
		if self.lint is not None:
			self.lo_expression = self.data[self.lint[0]].rna['rna'][:, self.lint[1]]

	def get_betas(self, expression=None, pfeatures=None, cfeatures=None):
		if expression is None:
			expression = self.expression
		if cfeatures is None:
			cfeatures = self.cfeatures
		if pfeatures is None:
			pfeatures = self.pfeatures
		self.betas = numpy.linalg.lstsq(
			numpy.concatenate([pfeatures, cfeatures], axis=1),
			expression, rcond=None)[0].reshape(-1, self.stateN, order='c')

	def refine_tads(self):
		self.get_features()
		self.get_betas()
		adjR2 = -1
		lo_adjR2 = -1
		best_adjR2 = adjR2
		best_tads = []
		for i in range(self.genomeN):
			best_tads.append(None)
		prev_R2 = -1
		for i in range(self.iterations):
			for j in range(self.genomeN):
				self.data[j].refine_tads(self.betas, i)
			self.get_features()
			self.get_betas()
			adjR2, mse = self.find_adjR2()
			if self.lint is not None:
				lo_adjR2, _ = self.find_adjR2(pfeatures=self.lo_pfeatures,
											  cfeatures=self.lo_cfeatures,
											  expression=self.lo_expression)
			else:
				lo_adjR2 = -1
			if adjR2 > best_adjR2:
				for j in range(self.genomeN):
					best_tads[j] = numpy.copy(self.data[j].tads)
				best_adjR2 = adjR2
			num_tads, mean_size, changed = self.tad_stats()
			self.logger.info(f"Iteration: {i+1}  {num_tads} TADs " + \
							 f"(mean size {mean_size:,.2f}, {changed} " + \
							 f"changed)   adj-R2: {adjR2*100:.2f}%%   " + \
							 f"LO adj-R2: {lo_adjR2*100:.2f}%%  " + \
							 f"MSE:{mse:,.4f}")
			if changed == 0:
				break
			prev_R2 = adjR2
		for j in range(self.genomeN):
			self.data[j].tads = numpy.copy(best_tads[j])

	def find_adjR2(self, pfeatures=None, cfeatures=None, expression=None):
		if expression is None:
			expr = numpy.copy(self.expression)
		else:
			expr = numpy.copy(expression)
		pred = self.find_predicted(pfeatures, cfeatures)
		mse = numpy.sum((pred - expr) ** 2)
		n = pred.shape[0]
		p = self.betas.size
		pred -= numpy.mean(pred)
		pred /= numpy.std(pred)
		expr -= numpy.mean(expr)
		expr /= numpy.std(expr)
		R2 = numpy.mean(pred * expr) ** 2
		aR2 = 1 - (1 - R2) * (n - 1) / (n - p - 1)
		return aR2, mse

	def find_predicted(self, pfeatures=None, cfeatures=None):
		if pfeatures is None:
			pfeatures = self.pfeatures
		if cfeatures is None:
			cfeatures = self.cfeatures
		pred = numpy.sum(pfeatures * self.betas[:1, :] +
						 cfeatures * self.betas[-1:, :], axis=1)
		return pred

	def tad_stats(self):
		num_tads = 0
		tad_size = 0
		for i in range(self.genomeN):
			num_tads += self.data[i].tads.shape[0]
			tad_size += numpy.sum(
				self.data[i].joint_coords[self.data[i].tads[:, 1] - 1] -
				self.data[i].joint_coords[self.data[i].tads[:, 0]])
			changed = numpy.sum(self.data[i].changed)
		return num_tads, tad_size / num_tads, changed

	def write_results(self):
		self.write_settings()
		self.write_betas()
		self.write_correlations()
		for i in range(self.genomeN):
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
		if self.lessone is not None:
			print("lessone = {}".format(", ".join(self.lessone.split(","))), file=output)
		print("shuffled: {}".format(self.shuffle), file=output)
		print("promoter_dist = {}".format(self.promoter_dist), file=output)
		print("initialization_dist = {}".format(self.initialization_dist), file=output)
		print("cre_dist = {}".format(self.cre_dist), file=output)
		print("iterations = {}".format(self.iterations), file=output)
		print("seed = {}".format(self.seed), file=output)
		output.close()

	def write_betas(self):
		output = open("{}_betas.txt".format(self.out_prefix), "w")
		print("Feature\tState\tBeta", file=output)
		for i in range(self.stateN):
			print("Promoter\t{}\t{}".format(i, self.betas[0, i]), file=output)
		for i in range(self.stateN):
			print("CRE\t{}\t{}".format(i, self.betas[-1, i]), file=output)
		output.close()

	def write_correlations(self):
		output = open("{}_correlations.txt".format(self.out_prefix), "w")
		print("Genome\tCelltype\tAdjR2", file=output)
		for i in range(self.genomeN):
			for j in range(self.data[i].cellN):
				adjR2, _ = self.find_adjR2(
					pfeatures=self.data[i].pfeatures[:, j, :],
					cfeatures=self.data[i].cfeatures[:, j, :],
					expression=self.data[i].rna['rna'][:, j])
				print(f"{self.genomes[i]}\t{self.data[i].celltypes[j]}\t{adjR2}",
					  file=output)
			adjR2, _ = self.find_adjR2(
				pfeatures=self.data[i].pfeatures.reshape(-1, self.stateN, order="c"),
				cfeatures=self.data[i].cfeatures.reshape(-1, self.stateN, order="c"),
				expression=self.data[i].rna['rna'].reshape(-1, order="c"))
			print(f"{self.genomes[i]}\tAll\t{adjR2}", file=output)
		pfeatures = []
		cfeatures = []
		expression = []
		for i in range(self.genomeN):
			pfeatures.append(self.data[i].pfeatures.reshape(
				-1, self.stateN, order="c"))
			cfeatures.append(self.data[i].cfeatures.reshape(
				-1, self.stateN, order="c"))
			expression.append(self.data[i].rna['rna'].reshape(-1, order="c"))
		adjR2, _ = self.find_adjR2(
			pfeatures=numpy.concatenate(pfeatures, axis=0),
			cfeatures=numpy.concatenate(cfeatures, axis=0),
			expression=numpy.concatenate(expression))
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
			print(f"\r{' '*80}\rLoading {self.genome} cCRE data",
				  end='', file=sys.stderr)
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

	def initialize_genome(self, initialization_dist, cre_dist, promoter_dist,
						  seed, shuffle):
		self.rng = numpy.random.default_rng(seed)
		self.shuffle = shuffle
		self.initialization_dist = initialization_dist
		self.cre_dist = cre_dist
		self.promoter_dist = promoter_dist
		if self.shuffle == 'tss':
			self.order = numpy.arange(self.tssN)
			self.rng.shuffle(self.order)
			self.rna['rna'] = self.rna['rna'][self.order, :]
		self.find_initial_features()

	def assign_promoter_states(self):
		"""Find the proportion of states in each promoter window"""
		if self.verbose >= 2:
			print(f"\r{' '*80}\rAssign states to {self.genome} promoters",
				  end='', file=sys.stderr)
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
							  dtype=numpy.float32)
		for i in range(self.cellN):
			s = self.state_cindices[i]
			e = self.state_cindices[i + 1]
			Pstates[:, i, :] = numpy.sum(Pstates2[:, s:e, :], axis=1) / (e - s)
		self.pfeatures = Pstates / (self.rna['end'] -
									self.rna['start']).reshape(-1, 1, 1)
		if self.shuffle == 'cre':
			self.order2 = numpy.arange(self.tssN)
			self.rng.shuffle(self.order2)
			self.pfeatures = self.pfeatures[self.order2, :, :]
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
			print(f"\r{' '*80}\r", end='', file=sys.stderr)

	def find_initial_features(self):
		self.assign_promoter_states()
		self.cfeatures = numpy.zeros((self.tssN, self.cellN, self.stateN),
									 dtype=numpy.float32)
		self.assign_CRE_states()
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
			if e == s or numpy.amin(self.rna['rna'][i, :]) == numpy.amax(
					self.rna['rna'][i, :]):
				continue
			pairs[s:e, 0] = i
			pairs[s:e, 1] = numpy.arange(tss_ranges[i, 0], tss_ranges[i, -1])
		pair_indices = numpy.r_[0, numpy.cumsum(numpy.bincount(pairs[:, 0],
															   minlength=self.tssN))]
		for i in range(self.tssN):
			s, e = pair_indices[i:(i+2)]
			if e == s:
				continue
			self.cfeatures[i, :, :] = numpy.sum(
				self.Cstates[pairs[s:e, 1], :, :], axis=0)

	def set_lo_mask(self, lo_mask):
		self.lo_mask = lo_mask
		self.lo_N = self.lo_mask.shape[0]

	def find_tss_cre_pairs(self):
		if self.verbose >= 2:
			print("\r{' '*80}\rFinding {self.genome} TSS-cCRE pairs",
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
			up_indices = numpy.searchsorted(coords, coords[TSSs] - self.cre_dist,
											side='right')
			for j, k in enumerate(TSSs):
				l = k
				while l > up_indices[j]:
					if joint[l - 1, 1] == 1:
						break
					l -= 1
				joint[k, 2] = l + js
			down_indices = numpy.searchsorted(coords, coords + self.cre_dist,
											  side='right')
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

	def find_nearest_gene_pairs(self):
		if self.verbose >= 2:
			print(f"\r{' '*80}\rFinding {self.genome} TSS-cCRE pairs using the nearest gene",
				  end='', file=sys.stderr)
		ng_pairs = numpy.zeros((self.creN, 2), dtype=numpy.int32) #will store the CRE index in the first column
		for i in range(self.chromN):
			js, je = self.joint_indices[i:(i+2)]
			CREs = numpy.where(self.joint_EP[js:je, 1] == 0)[0]
			TSSs = numpy.setdiff1d(numpy.arange(len(self.joint_coords[js:je])), CREs)
			coords_CREs = self.joint_coords[js:je][CREs]
			coords_TSSs = self.joint_coords[js:je][TSSs]
			indices_CREs = self.joint_EP[CREs + js, 0]
			indices_TSSs = self.joint_EP[TSSs + js, 0]
			for k, cre in enumerate(coords_CREs):
				#which genes are within allowed distance
				genes_within = numpy.where((coords_TSSs >= min(0, cre - self.cre_dist)) & (coords_TSSs <= cre + self.cre_dist))[0]
				if len(genes_within) >= 1:
					#which gene is closest
					dists = numpy.abs(cre - coords_TSSs[genes_within])
					shortest = numpy.where(dists == numpy.amin(dists))[0][0]
					gene_closest_indice = indices_TSSs[genes_within[shortest]]
				else:
					gene_closest_indice = -1
				#store coordinate of cre and store coordinate of its closest gene
				ng_pairs[k, :] = numpy.array([indices_CREs[k], gene_closest_indice])
		self.ng_pairs = ng_pairs
		if self.verbose >= 2:
			print(f"\r{' '*80}\r", file=sys.stderr)

	def find_features(self):
		for s, e in self.tads:
			TSSs = numpy.where(self.joint_EP[s:e, 1] == 1)[0] + s
			CREs = numpy.where(self.joint_EP[s:e, 1] == 0)[0] + s
			if CREs.shape[0] > 0:
				features = numpy.sum(self.Cstates[self.joint_EP[CREs, 0], :, :],
									 axis=0)
				self.cfeatures[self.joint_EP[TSSs, 0], :, :] = features
			else:
				self.cfeatures[self.joint_EP[TSSs, 0], :, :] = 0

	def find_features_ng(self):
		for TSS_index in numpy.arange(self.tssN):
			ngCRE = numpy.where(self.ng_pairs[:,1] == TSS_index)[0]
			CREs = self.ng_pairs[ngCRE, 0]
			if CREs.shape[0] > 0:
				features = numpy.sum(self.Cstates[CREs, :, :],
									axis=0)
				self.cfeatures[TSS_index, :, :] = features
			else:
				self.cfeatures[TSS_index, :, :] = 0

	def find_predicted(self, betas, pfeatures=None, cfeatures=None):
		if pfeatures is None:
			pfeatures = self.pfeatures[:, self.lo_mask, :]
		if cfeatures is None:
			cfeatures = self.cfeatures[:, self.lo_mask, :]
		predicted = numpy.sum(pfeatures * betas[0, :].reshape(1, 1, -1) +
							  cfeatures * betas[-1, :].reshape(1, 1, -1),
							  axis=2)
		return predicted

	def find_MSE(self, betas, expression=None, pfeatures=None, cfeatures=None):
		if expression is None:
			expression = self.rna['rna'][:, self.lo_mask]
		predicted = self.find_predicted(betas, pfeatures, cfeatures)
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
									betas[-1, :].reshape(1, 1, -1), axis=2).astype(
									numpy.float32)
			tstart, tend = self.rna_indices[c:(c+2)]
			pStateBetas = numpy.sum(self.pfeatures[tstart:tend, self.lo_mask, :] *
									betas[0, :].reshape(1, 1, -1),
									axis=2).astype(numpy.float32)
			RNA = numpy.copy(self.rna['rna'][tstart:tend, self.lo_mask])
			jstart, jend = self.joint_indices[c:(c+2)]
			joint = self.joint_EP[jstart:jend, :]
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
		MSE = self.find_MSE(betas)
		if self.verbose > 2:
			print(f"\r{' '*80}\r", end="", file=sys.stderr)
		return MSE

	def write_tads(self, out_prefix):
		output = open(f"{out_prefix}_{self.genome}_tads.bed", "w")
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
			print(f"{chrom}\t{start}\t{end}", file=output)
		output.close()

	def write_expression(self, betas, out_prefix):
		output = open(f"{out_prefix}_{self.genome}_expression.txt", "w")
		predicted = self.find_predicted(betas, self.pfeatures, self.cfeatures)
		temp = '\t'.join([f'eRP-{x}' for x in self.celltypes])
		print(f"TSS-Chr\tTSS-Coord\t{temp}", file=output)
		for j in range(self.tssN):
			gene = self.rna[j]
			temp = "\t".join(["{:.4f}".format(x) for x in predicted[j, :]])
			print(f"{gene['chr']}\t{gene['TSS']}\t{temp}", file=output)
		output.close()

	def write_order(self, out_prefix):
		if self.shuffle =='tss':
			output = open(f"{out_prefix}_{self.genome}_gene_order.txt", "w")
		else:
			output = open(f"{out_prefix}_{self.genome}_cre_order.txt", "w")
		for i in self.order:
			print(i, file=output)
		output.close()
		if self.shuffle == 'cre':
			output = open(f"{out_prefix}_{self.genome}_promoter_order.txt", "w")
			for i in self.order2:
				print(i, file=output)
			output.close()

	def write_pairs(self, betas, out_prefix):
		output = open(f"{out_prefix}_{self.genome}_pairs.txt", 'w')
		temp = ['chr', 'TSS', 'cre-start', 'cre-end']
		for ct in self.celltypes:
			temp.append(f'eRP-{ct}')
		print("\t".join(temp), file=output)
		for h in numpy.arange(self.tads.shape[0]):
			s, e = self.tads[h, :]
			temp = self.joint_EP[s:e, :]
			TSSs = temp[numpy.where(temp[:, 1] == 1)[0], 0]
			CREs = temp[numpy.where(temp[:, 1] == 0)[0], 0]
			if CREs.shape[0] > 0:
				eRPs = numpy.sum(self.Cstates[CREs, :, :] *
								 betas[-1, :].reshape(1, 1, -1), axis=2)
			for i in TSSs:
				TSS = self.rna[i]
				promoter = numpy.sum(self.fpeatures[i, :, :] *
									 betas[:1, :], axis=1)
				temp = "\t".join([f"{x}" for x in promoter])
				print(f"{TSS['chr']}\t{TSS['TSS']}\t{-1}\t{-1}\t{temp}",
					  file=output)
				for j, c in enumerate(CREs):
					CRE = self.cre[c]
					temp = "\t".join([f"{x}" for x in eRPs[j, :]])
					print(f"{TSS['chr']}\t{TSS['TSS']}\t{CRE['start']}" + \
						  f"\t{CRE['end']}\t{temp}", file=output)
		output.close()


if __name__ == "__main__":
	main()
