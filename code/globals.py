import os
import numpy as np
import matplotlib as mpl
import seaborn as sns
import yaml
from utils import create_dir

mpl.rcParams['font.family'] = 'Arial'
sns.set_context("paper", rc={
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5
}) 
np.random.seed(0)

TILE_LEN = 53
TILE_SPACING = 10
AMINO_ACIDS = ['R', 'K', 'D', 'E', 'Q', 'N', 'H', 'S', 'T', 'Y', 'C', 'W', 'M', 'A', 'I', 'L', 'F', 'V', 'P', 'G']

FEATURE_DICT = {
    'aliphatics':['I','V','L','A'],
    'aromatics':['W','F','Y'],
    'branching':['V','I','T'],
    'charged':['K','R','H','D','E'],
    'negatives':['D','E'],
    'phosphorylatables':['S','T','Y'],
    'polars':['R','K','D', 'E', 'Q', 'N', 'Y'],
    'hydrophobics':['W','F','L','V', 'I', 'C', 'M'],
    'positives':['K','R','H'],
    'sulfurcontaining':['M','C'],
    'tinys':['G','A','S','P']
 }

AA_DICT = {}
for feature, amino_acids in FEATURE_DICT.items():
    for aa in amino_acids:
        if aa not in AA_DICT:
            AA_DICT[aa] = []
        AA_DICT[aa].append(feature)

with open(os.path.join(os.path.dirname(__file__), "config.yml"), "r") as _f:
    _config = yaml.safe_load(_f)
    
SAVE_FIGURES = _config.get("save_figures", True)
FIG_PARAMS = dict(bbox_inches='tight',transparent=True,pad_inches=0)

IN_DIR = '../data/01-input'
METADATA_DIR = f'{IN_DIR}/01-metadata'
GENOME_DIR = f'{IN_DIR}/02-genomes'
EXPERIMENT_DIR = f'{IN_DIR}/03-experiments'

OUT_DIR = '../data/02-output'
VIRUS_DIR = f'{OUT_DIR}/02-viruses'
TILE_DIR = f'{OUT_DIR}/03-tiles'
INFERENCE_DIR = f'{OUT_DIR}/04-inference'
LIBRARY_DIR = f'{OUT_DIR}/05-library'
ANALYSIS_DIR = f'{OUT_DIR}/06-analysis'
CONSTANS_DIR = f'{OUT_DIR}/07-constans'

FIGURE_DIR = f'{OUT_DIR}/01-figures'
for output_dir in [FIGURE_DIR]:
    create_dir(output_dir)