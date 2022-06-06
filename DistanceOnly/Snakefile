import gzip
import numpy

###### User-defined variables ######

LINK_DICT = {
  "mm10_rna.txt"             : "http://usevision.org/data/mm10/rnaHtseqCountsall_withcoordinates.0.txt",
  "mm10_state.txt"           : "http://usevision.org/data/mm10/ideasJointMay2021/S3V2_IDEAS_mm10_r3_withHg38Mm10prior.state",
  "mm10_cre.txt"             : "http://usevision.org/data/mm10/ideasJointMay2021/S3V2_IDEAS_mm10_ccre2.cCRE.M.notall0.bed",
  "hg38_rna.txt"             : "http://usevision.org/data/hg38/RNA/Oct2021/cntsFeb21v3.tab",
  "hg38_state.txt"           : "http://usevision.org/data/hg38/IDEASstates/ideasJointMay2021/S3V2_IDEAS_hg38_r3_withHg38Mm10prior.state",
  "hg38_cre.txt"             : "http://usevision.org/data/hg38/IDEASstates/ideasJointMay2021/S3V2_IDEAS_hg38_ccre2.cCRE.M.notall0.rmallNEU.bed"
}

CRE_ITERS = 50
TAD_ITERS = 50
PROMOTER_DIST = 100
CRE_DIST = 1000000
REPS = 1


###### Snakemake-defined variables ######

FILES = list(LINK_DICT.keys())
SPECIES = ["mm10", "hg38"]
FEATURES = ['cre', 'promoter', 'both']
SPECIES_PAIR = {
  "mm10": "hg38",
  "hg38": "mm10"
}

FILE_SET = "|".join(FILES)
SPECIES_SET = "|".join(SPECIES)
FEATURE_SET = "|".join(FEATURES)

RNG = numpy.random.default_rng()

def all_inputs(wc):
  treat = []
  control = []
  for species in SPECIES:
    with checkpoints.get_celltypes.get(species=species).output[0].open() as f:
      CTs = f.readline().rstrip().split()
    for ct in CTs:
      for feat in FEATURES:
        if feat == 'promoter':
          R = 1
        else:
          R = REPS
        for i in range(R):
          treat.append("Results_{0}/{0}_{1}/{2}_{3}_betas.txt".format(
                       species, ct, i, feat))
          control.append("Results_{0}/{0}_{1}/{2}_control_{3}_betas.txt".format(
                         species, ct, i, feat))
  return treat + control

rule all:
  input:
    all_inputs


######## Download and preprocess text files into numpy array files ########

rule download_data:
  output:
    "data/{file}"
  params:
    link=lambda wildcards: LINK_DICT[wildcards.file]
  wildcard_constraints:
    file=FILE_SET
  shell:
    """
    wget -O {output} {params.link}
    """

rule preprocess_mm10_rna:
  input:
    "data/mm10_rna.txt"
  output:
    "data/mm10_rna_all.npy"
  conda:
    "envs/general.yaml"
  shell:
    """
    bin/mm10_rna_to_npy.py -r {input} -o {output}
    """

rule preprocess_hg38_rna:
  input:
    rna="data/hg38_rna.txt",
    ids="data/hg38_ensembl_IDs.txt"
  output:
    "data/hg38_rna_all.npy"
  conda:
    "envs/general.yaml"
  shell:
    """
    bin/hg38_rna_to_npy.py -r {input.rna} --ids {input.ids} -o {output}
    """

rule split_rna:
  input:
    "data/{genome}_rna_all.npy"
  output:
   diff="data/{genome}_rna_diff.npy",
   nodiff="data/{genome}_rna_nodiff.npy",
   plot="plots/{genome}_rna_split.pdf"
  params:
    prefix="data/{genome}_rna"
  conda:
    "envs/general.yaml"
  shell:
    """
    bin/split_rna.py -r {input} -p {output.plot} -o {params.prefix}
    """

rule preprocess_states:
  input:
    "data/{species}_state.txt"
  output:
    "data/{species}_state.npy"
  wildcard_constraints:
    species=SPECIES_SET
  conda:
    "envs/general.yaml"
  shell:
    """
    head -n 1 {input} | sed -e 's/T_CD/CD/g' > {output}.tmp
    tail -n +2 {input} >> {output}.tmp
    bin/states_to_npy.py -s {output}.tmp -o {output}
    rm {output}.tmp
    """

rule preprocess_CREs:
  input:
    "data/{species}_cre.txt"
  output:
    "data/{species}_cre.npy"
  wildcard_constraints:
    species=SPECIES_SET
  conda:
    "envs/general.yaml"
  shell:
    """
    bin/CREs_to_npy.py -c {input} -o {output}
    """

checkpoint get_celltypes:
  input:
    states="data/{species}_state.npy",
    rna="data/{species}_rna_all.npy"
  output:
    "data/{species}_celltypes.txt"
  wildcard_constraints:
    species=SPECIES_SET
  conda:
    "envs/general.yaml"
  shell:
    """
    bin/get_shared_celltypes.py {input} > {output}
    """



######## Run training of model ########

def get_params(wc):
  params = {}
  params['--output'] = "Results_{}/{}_{}/{}_{}".format(wc.species, wc.species,
                                                       wc.ct, wc.label, wc.feat)
  seed = int(RNG.random() * 10000000)
  label = wc.label.split('_')
  if len(label) == 1:
    rep = label[0]
  else:
    rep, control = label
    params['--shuffle'] = ''
  if rep != "0":
    params['--rand-init'] = ''
  params['--genome'] = "{} {}".format(wc.species, SPECIES_PAIR[wc.species])
  params['--lessone'] = "{},{}".format(wc.species, wc.ct)
  params['--seed'] = seed
  if wc.feat == "cre":
    params['--promoter-dist'] = 0
    params['--cre-dist'] = CRE_DIST
  elif wc.feat == "promoter":
    params['--promoter-dist'] = PROMOTER_DIST
    params['--cre-dist'] = 0
  else:
    params['--promoter-dist'] = PROMOTER_DIST
    params['--cre-dist'] = CRE_DIST
  params['--beta-iter'] = CRE_ITERS
  params['--tad-iter'] = TAD_ITERS
  params['--npy'] = ''
  params['--verbose'] = 1
  return [' '.join(["{} {}".format(k, v) for k, v in params.items()]),
          str(seed)]

rule train_model:
  input:
    rna1="data/{species}_rna_all.npy",
    states1="data/{species}_state.npy",
    cre1="data/{species}_cre.npy",
    rna2=lambda wildcards: expand("data/{species2}_rna_all.npy",
                                  species2=SPECIES_PAIR[wildcards.species]),
    states2=lambda wildcards: expand("data/{species2}_state.npy",
                                     species2=SPECIES_PAIR[wildcards.species]),
    cre2=lambda wildcards: expand("data/{species2}_cre.npy",
                                  species2=SPECIES_PAIR[wildcards.species])
  output:
    betas="Results_{species}/{species}_{ct}/{label}_{feat}_betas.txt",
    erp="Results_{species}/{species}_{ct}/{label}_{feat}_{species}_pairs.npy",
    expression="Results_{species}/{species}_{ct}/{label}_{feat}_{species}_expression.npy",
    correlations="Results_{species}/{species}_{ct}/{label}_{feat}_correlations.txt",
    settings="Results_{species}/{species}_{ct}/{label}_{feat}_settings.txt",
    seed="Results_{species}/{species}_{ct}/{label}_{feat}_seed.txt"
  wildcard_constraints:
    species=SPECIES_SET,
    feat="cre|both",
    label="\d+|\d+_control"
  params:
    get_params
  conda:
    "envs/general.yaml"
  shell:
    """
    bin/JointDistance.py \
      --rna {input.rna1} {input.rna2} \
      --state {input.states1} {input.states2} \
      --cre {input.cre1} {input.cre2} \
      {params[0][0]}
    echo {params[0][1]} > {output.seed}
    """

rule train_promoter_model:
  input:
    rna1="data/{species}_rna_all.npy",
    states1="data/{species}_state.npy",
    cre1="data/{species}_cre.npy",
    rna2=lambda wildcards: expand("data/{species2}_rna_all.npy",
                                  species2=SPECIES_PAIR[wildcards.species]),
    states2=lambda wildcards: expand("data/{species2}_state.npy",
                                     species2=SPECIES_PAIR[wildcards.species]),
    cre2=lambda wildcards: expand("data/{species2}_cre.npy",
                                  species2=SPECIES_PAIR[wildcards.species])
  output:
    betas="Results_{species}/{species}_{ct}/{label}_{feat}_betas.txt",
    expression="Results_{species}/{species}_{ct}/{label}_{feat}_{species}_expression.npy",
    correlations="Results_{species}/{species}_{ct}/{label}_{feat}_correlations.txt",
    settings="Results_{species}/{species}_{ct}/{label}_{feat}_settings.txt",
    seed="Results_{species}/{species}_{ct}/{label}_{feat}_seed.txt"
  wildcard_constraints:
    species=SPECIES_SET,
    feat="promoter",
    label="\d+|\d+_control"
  params:
    get_params
  conda:
    "envs/general.yaml"
  shell:
    """
    bin/JointDistance.py \
      --rna {input.rna1} {input.rna2} \
      --state {input.states1} {input.states2} \
      --cre {input.cre1} {input.cre2} \
      {params[0][0]}
    echo {params[0][1]} > {output.seed}
    """


######## Plot figures ########

def plot_correlation_inputs(wc):
  with checkpoints.get_celltypes.get(species=wc.species).output[0].open() as f:
    CTs = f.readline().rstrip().split()
  treat = []
  control = []
  for ct in CTs:
    for gene in GENES:
      for feat in FEATURES:
        treat.append("Results_{0}/{0}_{1}_{2}_{3}/0_statistics.txt".format(
                     wc.species, gene, ct, feat))
        control.append("Results_{0}/{0}_{1}_{2}_{3}/0_control_statistics.txt".format(
                       wc.species, gene, ct, feat))
  return {"treat": treat, "control": control}

rule plot_correlations:
  input:
    unpack(plot_correlation_inputs)
  output:
    "plots/{species}_correlations.pdf"
  params:
    datadir = "Results_{species}"
  conda:
    "envs/general.yaml"
  shell:
    """
    bin/plot_correlations.py {params.datadir} {output}
    """

def plot_beta_inputs(wc):
  with checkpoints.get_celltypes.get(species=wc.species).output[0].open() as f:
    CTs = f.readline().rstrip().split()
  treat = []
  control = []
  for ct in CTs:
    for gene in GENES:
      for feat in FEATURES:
        treat.append("Results_{0}/{0}_{1}_{2}_{3}/0_betas.txt".format(
                     wc.species, gene, ct, feat))
        control.append("Results_{0}/{0}_{1}_{2}_{3}/0_control_betas.txt".format(
                       wc.species, gene, ct, feat))
  return {"treat": treat, "control": control}

rule plot_betas:
  input:
    unpack(plot_beta_inputs)
  output:
    "plots/{species}_betas.pdf"
  params:
    datadir = "Results_{species}"
  conda:
    "envs/general.yaml"
  shell:
    """
    bin/plot_betas.py {params.datadir} {output}
    """

def plot_reproducibility_inputs(wc):
  with checkpoints.get_celltypes.get(species=wc.species).output[0].open() as f:
    CTs = f.readline().rstrip().split()
  treat = []
  control = []
  for ct in CTs:
    for i in range(REPS):
      for feat in FEATURES:
        treat.append("Results_{0}/{0}_all_{1}_{2}/{3}_betas.txt".format(
                     wc.species, ct, feat, i))
        control.append("Results_{0}/{0}_all_{1}_{2}/{3}_control_betas.txt".format(
                       wc.species, ct, feat, i))
    rna = "data/{}_rna_all.npy".format(wc.species)
    state = "data/{}_state.npy".format(wc.species)
    cre = "data/{}_cre.npy".format(wc.species)
  return {"treat": treat, "control": control, "rna": rna, "state": state, "cre": cre}

rule plot_reproducibility:
  input:
    unpack(plot_reproducibility_inputs)
  output:
    "plots/{species}_reproducibility.pdf"
  params:
    prefix = "Results_{species}/{species}",
    species = "{species}"
  wildcard_constraints:
    rep="\d+"
  conda:
    "envs/general.yaml"
  shell:
    """
    bin/plot_reproducibility.py {params.species} {params.prefix} \
      {input.state} {input.rna} {input.cre} {output}
    """
