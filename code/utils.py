from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from Bio import Entrez
from tqdm import tqdm
import os
from ete3 import NCBITaxa, Tree
from matplotlib_venn import venn2_unweighted,venn2_circles
from collections import Counter
from Bio.Seq import Seq
import numpy as np
from tqdm import trange
from Bio.Seq import Seq
from dnachisel import *
from matplotlib_venn import venn3_unweighted, venn3_circles
import matplotlib.patches as patches
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib as mpl
from scipy.optimize import curve_fit, root_scalar
from scipy.stats import pearsonr, spearmanr
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
import math
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def calculate_mad(series):
    median = series.median()
    return np.median(np.abs(series - median))

def shannon_entropy(column):
    length = len(column)
    freqs = Counter(column)
    entropy = -sum((count / length) * math.log2(count / length) for count in freqs.values() if count > 0)
    return entropy

def calculate_entropy_from_fasta(fasta_path):
    sequences = [str(record.seq) for record in SeqIO.parse(fasta_path, "fasta")]
    seq_len = len(sequences[0])
    entropies = []
    for i in range(seq_len):
        column = [seq[i] for seq in sequences]
        entropies.append(shannon_entropy(column))
    
    return entropies

def entropy_from_dataframe(df, column_name):
    sequences = df[column_name].dropna().astype(str).tolist()
    seq_len = len(sequences[0])
    entropies = []
    for i in range(seq_len):
        position_residues = [seq[i] for seq in sequences]
        entropies.append(shannon_entropy(position_residues))
    return entropies

def fasta_to_dataframe(fasta_path):
    records = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        records.append({
            "id": record.id,
            "description": record.description,
            "sequence": str(record.seq)
        })
    return pd.DataFrame(records)

def count_fasta_sequences(file_path):
    count = 0
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('>'):
                count += 1
    return count

def max_overlap(seq1, seq2):
    max_len = 0
    for i in range(len(seq1)):
        if seq2.startswith(seq1[i:]):
            max_len = len(seq1) - i
            break
    return max_len

def compute_overlap_matrix(sequences):
    n = len(sequences)
    overlap_matrix = [[0] * n for _ in range(n)]
    
    for i, seq1 in tqdm(enumerate(sequences),total=len(sequences)):
        for j, seq2 in enumerate(sequences):
            if i != j: # diagonal
                overlap_matrix[i][j] = max_overlap(seq1, seq2)
    
    return overlap_matrix

def calculate_positional_entropy(sequences):
    sequence_length = len(sequences[0])
    total_entropy = 0.0
    for position in range(sequence_length):
        column = [seq[position] for seq in sequences]
        counts = Counter(column)
        total = len(column)
        entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
        total_entropy += entropy

    return total_entropy

def calculate_entropy(sequence):
    freq = Counter(sequence)
    total_amino_acids = len(sequence)
    entropy = 0
    for count in freq.values():
        probability = count / total_amino_acids
        entropy -= probability * math.log(probability, 2)
    return entropy

def load_interproscan(filepath):
    df = pd.read_csv(filepath, sep='\t', header=None)
    df.columns = [
        "Protein_ID", "Sequence_MD5", "Sequence_Length", "Analysis",
        "Signature_Accession", "Signature_Description", "Start", "End",
        "Score", "Status", "Date", "InterPro_Accession", "InterPro_Description",
        "GO_Terms", "Pathway"
    ]
    return df

def filter_proteins(interpro_df, go_df, keyword):
    keyword_mask = interpro_df['InterPro_Description'].str.contains(keyword, case=False, na=False)
    go_mask = interpro_df['Protein_ID'].isin(
        go_df[go_df['Description'].str.contains(keyword, case=False, na=False)]['protein_ID'].unique()
    )
    filtered = interpro_df[keyword_mask | go_mask]
    filtered["Score"] = pd.to_numeric(filtered["Score"], errors="coerce")
    return filtered.loc[filtered.groupby("Protein_ID")["Score"].idxmin()]

def prepare_tmp_df(library_df, protein_ids):
    tmp = library_df[library_df['protein_ID'].isin(protein_ids)].copy()
    len_map = tmp.groupby('protein_ID')['tile_count'].agg(lambda x: int(max(x)) + 1).to_dict()
    tmp['protein_len'] = tmp['protein_ID'].map(len_map)
    tmp['norm_pos'] = tmp['tile_count'].astype(int) / tmp['protein_len']
    return tmp

def plot_panel(ax, tmp_df, positional_file, title, color_threshold=4.9, density_downsample=True, max_points=5000):
    if density_downsample and len(tmp_df) > max_points:
        # calculate density in (norm_pos, zscore) space
        coords = np.vstack([tmp_df['norm_pos'], tmp_df['zscore']])
        kde = gaussian_kde(coords)
        tmp_df['density'] = kde(coords)
        tmp_df['inv_density'] = 1 / tmp_df['density']
        tmp_df['prob'] = tmp_df['inv_density'] / tmp_df['inv_density'].sum()
        
        tmp_df = tmp_df.sample(n=max_points, weights='prob', random_state=42)

    sns.scatterplot(
        data=tmp_df.query(f'zscore < {color_threshold}'), x='norm_pos', y='zscore',
        edgecolor=None, alpha=1.0, s=7.5, ax=ax, color='lightgrey'
    )
    sns.scatterplot(
        data=tmp_df.query(f'zscore >= {color_threshold}'), x='norm_pos', y='zscore',
        edgecolor=None, alpha=1.0, s=7.5, ax=ax, color='skyblue'
    )
    if positional_file:
        positional_df = pd.read_csv(positional_file).query('is_activator == True')
        sns.scatterplot(
            data=tmp_df[tmp_df['tile_ID'].isin(positional_df['tile_ID'])], x='norm_pos', y='zscore',
            edgecolor=None, alpha=1.0, s=7.5, ax=ax, color='tomato'
        )
    ax.axhline(y=color_threshold, linestyle='--', color='grey', linewidth=0.5)
    ax.set(xlim=(-0.1, 1.1), xlabel=None, ylabel=None, title=title)

def find_motifs(sequences,k=2):
    motif_counts = {}
    for index, sequence in enumerate(sequences):
        seq_obj = Seq(sequence)
        kmer_counts = {}

        for i in range(len(seq_obj) - k + 1):
            kmer = str(seq_obj[i:i+k])
            if kmer in kmer_counts:
                kmer_counts[kmer] += 1
            else:
                kmer_counts[kmer] = 1
        if kmer_counts:
            motif_counts[index] = kmer_counts
    return motif_counts

def amino_acid_frequency_table(sequences):
    max_length = max(len(seq) for seq in sequences)
    position_counters = [Counter() for _ in range(max_length)]
    for seq in sequences:
        for i, amino_acid in enumerate(seq):
            position_counters[i][amino_acid] += 1
    frequency_table = []
    for counter in position_counters:
        total = sum(counter.values())
        frequency_table.append({
            amino_acid: count / total
            for amino_acid, count in counter.items()
        })
    return frequency_table

def calculate_amino_acid_frequency(sequence):
    frequency = {}
    total_length = len(sequence)
    for amino_acid in sequence:
        if amino_acid in frequency:
            frequency[amino_acid] += 1
        else:
            frequency[amino_acid] = 1
    for amino_acid in frequency:
        frequency[amino_acid] /= total_length
    return frequency

def calculate_frequencies_for_sequences(sequences):
    frequencies = {}
    for index, sequence in enumerate(sequences):
        frequencies[f"Protein_{index+1}"] = calculate_amino_acid_frequency(sequence)
    return frequencies

def lookup_motif(curr_motif, experimental_tiles):
    counter = 0
    for experimental_tile in experimental_tiles:
        if curr_motif in experimental_tile:
            counter += 1
    return counter

def fasta_to_df(fasta):
    fasta_records = []
    fasta_sequences = SeqIO.parse(fasta, 'fasta')
    for record in fasta_sequences:
        fasta_records.append({
            'id': record.id,
            'description': record.description,
            'seq': str(record.seq)
        })
    df = pd.DataFrame(fasta_records)
    return df

def protein_to_dna(protein_sequence):
    codon_table = {
        'A': 'GCT', 
        'R': 'CGT', 
        'N': 'AAT', 
        'D': 'GAT', 
        'C': 'TGT',
        'Q': 'CAA', 
        'E': 'GAA', 
        'G': 'GGT', 
        'H': 'CAT', 
        'I': 'ATT',
        'L': 'TTA', 
        'K': 'AAA', 
        'M': 'ATG', 
        'F': 'TTT', 
        'P': 'CCT',
        'S': 'TCT', 
        'T': 'ACT', 
        'W': 'TGG', 
        'Y': 'TAT', 
        'V': 'GTT',
        '*': 'TAA'
    }
    dna_sequence = ''
    for amino_acid in protein_sequence:
        dna_sequence += codon_table.get(amino_acid, '')
    return dna_sequence

def codon_optimize(seq,aa_dict):
    problem = DnaOptimizationProblem(
        sequence=seq,
        constraints=[
            AvoidPattern(str(12) + "xA"),
            AvoidPattern(str(12) + "xT"),
            AvoidPattern(str(6) + "xC"),
            AvoidPattern(str(6) + "xG"),
            AvoidHairpins(stem_size=10, hairpin_window=200),
            AvoidPattern("BsaI_site"),
            AvoidPattern("BsmBI_site"),
            AvoidPattern("PaqCI_site"),
            EnforceGCContent(mini=0.35, maxi=0.75, window=200),
            EnforceTranslation(),
        ],
        objectives=[CodonOptimize(codon_usage_table=aa_dict,method='use_best_codon')]
    )
    problem.max_random_iters = 50_000
    problem.resolve_constraints()
    problem.optimize()
    final_sequence = problem.sequence
    return final_sequence

def create_protein_tiles(protein_sequence, tile_size=40, overlap_size=10):
    tiles = []
    sequence_length = len(protein_sequence)
    for i in range(0, sequence_length - tile_size + 1, tile_size - overlap_size):
        tile = protein_sequence[i:i + tile_size]
        tiles.append(tile)
    tiles.append(protein_sequence[-tile_size:])
    return tiles

def seqToTiles(seq, tileLength, tileSpacing): 
    sequenceLength = len(seq)
    tilesList = []
    count = 0
    for i in range(0, sequenceLength, tileSpacing):
        if (i + tileLength) >= sequenceLength - 3:
            tilesList.append((count, seq[-(tileLength+3):-3]))
            return tilesList
        else:
            tilesList.append((count, seq[i:i+tileLength]))
            count += 1

def remove_sites(seq):
    problem = DnaOptimizationProblem(
        sequence=seq,
        constraints=[
            AvoidPattern("BsaI_site"),
            AvoidPattern("BsmBI_site"),
            AvoidPattern("PaqCI_site"),
            EnforceTranslation(),
        ],
    )
    problem.max_random_iters = 50_000
    problem.resolve_constraints()
    problem.optimize()
    final_sequence = problem.sequence
    return final_sequence

def one_hot_encode_protein(seq):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_index = {aa: i for i, aa in enumerate(amino_acids)}
    one_hot_matrix = np.zeros((len(seq), len(amino_acids)))
    for i, aa in enumerate(seq):
        if aa in aa_to_index:
            one_hot_matrix[i, aa_to_index[aa]] = 1
        else:
            raise ValueError(f"Invalid amino acid '{aa}' found in sequence.")
    return one_hot_matrix.flatten().astype(int)

def hamming_distance(array1, array2):
    return np.sum(array1 != array2)

def select_maximally_different_arrays(arrays, num_selected):
    selected_arrays = []
    selected_indices = []
    first_array_index = np.argmax(np.sum(arrays, axis=1))
    first_array = arrays[first_array_index]
    selected_arrays.append(first_array)
    selected_indices.append(first_array_index)
    for _ in tqdm(range(num_selected - 1)):
        dissimilarities = [np.sum([hamming_distance(array, selected_array) for selected_array in selected_arrays]) for array in arrays]
        selected_array_index = np.argmax(dissimilarities)
        while selected_array_index in selected_indices:
            dissimilarities[selected_array_index] = 0 
            selected_array_index = np.argmax(dissimilarities)
        selected_arrays.append(arrays[selected_array_index])
        selected_indices.append(selected_array_index)
    return selected_arrays, selected_indices

def find_row_in_matrix(matrix, row):
    matches = np.where((matrix == row).all(axis=1))[0]
    return matches

# def sigmoid_fit(x, L, x0, k, b):
#     return L / (1 + np.exp(-k*(x - x0))) + b

def sigmoid_fit(x, L, k, x0, b):
    exponent = -k * (x - x0)
    exponent = np.clip(exponent, -500, 500)  # avoid overflow
    return L / (1 + np.exp(exponent)) + b

def get_taxonomic_family(taxid):
    try:
        lineage = ncbi.get_lineage(taxid)
        ranks = ncbi.get_rank(lineage)
        names = ncbi.get_taxid_translator(lineage)

        for lin_taxid in lineage:
            if ranks[lin_taxid] == 'family':
                return names[lin_taxid]
        return "Family not found"
    except Exception as e:
        return f"Error: {e}"
    
def tile_to_residue_profile(predictions, tile_size, seq_len, stride=1):
    pred_sum = np.zeros(seq_len)
    pred_count = np.zeros(seq_len)

    for i, pred in enumerate(predictions):
        start = i * stride
        end = min(start + tile_size, seq_len)
        pred_sum[start:end] += pred
        pred_count[start:end] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        residue_profile = np.true_divide(pred_sum, pred_count)
        residue_profile[np.isnan(residue_profile)] = 0

    return residue_profile

def fit_sigmoid(x_data,y_data):
    def sigmoid(x, L, x0, k, b):
        return L / (1 + np.exp(-k * (x - x0))) + b
    p0 = [max(y_data), np.median(x_data), 1, min(y_data)]
    popt, _ = curve_fit(sigmoid, x_data, y_data, p0, maxfev=10000)
    L, x0, k, b = popt
    x_fit = np.linspace(min(x_data), max(x_data), 200)
    y_fit = sigmoid(x_fit, *popt)
    y_pred = sigmoid(x_data, *popt)
    residuals = y_data - y_pred
    residual_std = np.std(residuals)
    y_upper = y_fit + residual_std
    y_lower = y_fit - residual_std

    def y_lower_func(x):
        return sigmoid(x, *popt) - residual_std

    result = root_scalar(y_lower_func, bracket=[min(x_fit), max(x_fit)], method='brentq')
    x_intercept = result.root if result.converged else None
    y_midpoint = L / 2 + b
    y_upper_at_midpoint = y_midpoint + residual_std

    def func_upper_bound(x):
        return sigmoid(x, L, x0, k, b) - y_upper_at_midpoint

    x_upper_bound_result = root_scalar(func_upper_bound, bracket=[x0, max(x_fit)], method='brentq')
    x_upper_bound = x_upper_bound_result.root if x_upper_bound_result.converged else None

    print(x_upper_bound)
    return x_fit, y_fit, y_lower, y_upper, x_intercept, residual_std, x0, y_midpoint, x_upper_bound

def get_taxid_from_accession(accession):
    handle = Entrez.esearch(db="nuccore", term=accession)
    record = Entrez.read(handle)
    handle.close()
    if record["IdList"]:
        uid = record["IdList"][0]
        handle = Entrez.esummary(db="nuccore", id=uid)
        summary = Entrez.read(handle)
        handle.close()
        taxid = summary[0]["TaxId"]
        return int(taxid)
    return None

def parse_metapredict(data_file):
    with open(data_file, 'r') as f:
        lines = f.readlines()

        parsed_data = {}
        for line in lines:
            parts = line.strip().split(',') 
            if len(parts) < 2:
                continue 
            seq_id = parts[0]
            try:
                values = list(map(float, parts[2:]))
            except ValueError:
                continue
            parsed_data[seq_id] = values
    return pd.DataFrame([parsed_data]).T.reset_index()

def make_long_df(df, list_col=0, n_bins=50):
    records = []

    for _, row in df.iterrows():
        values = row[list_col]
        seq_len = len(values)
        if seq_len == 0:
            continue

        norm_pos = np.linspace(0, 1, seq_len, endpoint=False)
        records.extend({'norm_pos': x, 'value': y} for x, y in zip(norm_pos, values))

    long_df = pd.DataFrame(records)
    long_df['bin'] = pd.cut(long_df['norm_pos'], bins=n_bins)

    smoothed = long_df.groupby('bin').agg(
        norm_pos=('norm_pos', 'mean'),
        mean_value=('value', 'mean'),
        sd=('value', 'std')
    ).dropna().reset_index(drop=True)

    return smoothed

def create_dir(path):
    os.makedirs(path, exist_ok=True)

ncbi = NCBITaxa()

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

SAVE_FIGURES = True
FIG_PARAMS = dict(bbox_inches='tight',transparent=True,pad_inches=0)

IN_DIR = '/data/01-INPUT'
METADATA_DIR = f'{IN_DIR}/01-metadata'
GENOME_DIR = f'{IN_DIR}/02-genomes'
EXPERIMENT_DIR = f'{IN_DIR}/03-experiments'

OUT_DIR = '/data/02-OUTPUT'
VIRUS_DIR = f'{OUT_DIR}/02-viruses'
TILE_DIR = f'{OUT_DIR}/03-tiles'
INFERENCE_DIR = f'{OUT_DIR}/04-inference'
LIBRARY_DIR = f'{OUT_DIR}/05-library'
ANALYSIS_DIR = f'{OUT_DIR}/06-analysis'
CONSTANS_DIR = f'{OUT_DIR}/07-constans'

FIGURE_DIR = f'/results/01-figures'
for output_dir in [FIGURE_DIR]:
    create_dir(output_dir)