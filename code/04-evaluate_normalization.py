import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from ete3 import NCBITaxa
from utils import get_taxid_from_accession, calculate_mad
from globals import (
    EXPERIMENT_DIR, LIBRARY_DIR, TILE_DIR, ANALYSIS_DIR,
    FIGURE_DIR, SAVE_FIGURES, FIG_PARAMS,
)

_config = json.loads(os.environ.get("SCRIPT_CONFIG", "{}"))
SCRIPT_ID = '04'

import_empirical_data = _config.get("import_empirical_data", False)
out_file = f'{EXPERIMENT_DIR}/magnetic_sorting_data.csv'
if import_empirical_data:
    ncbi = NCBITaxa()
    dfs = []
    for data_file in glob(f'{EXPERIMENT_DIR}/magnets/*_compiled_data.csv'):
        dataset_id = data_file.split('/')[-1].replace('_compiled_data.csv','')
        tmp_df = pd.read_csv(data_file,index_col=0)
        tmp_df['dataset'] = dataset_id
        dfs.append(tmp_df)
    data_df = pd.concat(dfs).reset_index(drop=True)
    large_lib = pd.read_csv(f'{LIBRARY_DIR}/eLW044-fLW132.csv')
    mapper = dict(zip(large_lib['library_ID'],large_lib['tile_ID']))
    data_df['tileID'] = data_df['tileID'].replace(mapper)
    seq_df = pd.concat([
        pd.read_csv(f'{TILE_DIR}/viral_tiles.csv'),
        pd.read_csv(f'{LIBRARY_DIR}/viral_controls.csv')])[['tile_ID','tile']]
    seq_df['tile'] = seq_df['tile'].apply(lambda x: x[:-1] + x[-1].replace('*',''))
    data_df = (data_df
               .rename(columns={'tileID':'tile_ID'})
               .merge(seq_df,on='tile_ID'))
    protein_df = pd.read_csv(f'{TILE_DIR}/viral_proteins_clustered90.csv')
    id_map = dict(zip(protein_df['protein_ID'],protein_df['accession']))
    data_df['protein_id'] = data_df['tile_ID'].apply(lambda x: '-'.join(x.split('-')[:-1]))
    data_df['ncbi_id'] = data_df['protein_id'].replace(id_map)

    genbank_accessions = list(data_df['ncbi_id'].unique())
    if not os.path.isfile(f'{ANALYSIS_DIR}/tax_ids.csv'):
        ncbi_ids = []
        for acc in tqdm(genbank_accessions):
            if 'NC' in acc:
                ncbi_ids.append(get_taxid_from_accession(acc))
        tax_ids = [taxid for taxid in ncbi_ids if taxid is not None]
        pd.DataFrame(tax_ids).to_csv(f'{ANALYSIS_DIR}/tax_ids.csv',index=False)

        tree = ncbi.get_topology(tax_ids)
        tree.write(format=1, outfile=f'{ANALYSIS_DIR}/taxonomical_tree.nwk')
    else:
        tax_ids = pd.read_csv(f'{ANALYSIS_DIR}/tax_ids.csv')["0"].to_list()

    tax_mapper = dict(zip(genbank_accessions,tax_ids))
    data_df['tax_id'] = data_df['ncbi_id'].map(tax_mapper)
    data_df.to_csv(out_file,index=False)
else:
    data_df = pd.read_csv(out_file)

data_df['ratio_mean'] = data_df[['ratio_rep1','ratio_rep2']].mean(axis=1)
tmp_df = data_df.drop(columns=[
    'log_cov_rep1', 'log_cov_rep2', 'log_cov_rep3',
    'protein_id', 'ncbi_id', 'tax_id'
]).copy()

fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(6, 3.5), sharex='col')

datasets = {
    "250": {"filter": None},
    "500": {"filter": "ratio_rep2 < 10"},
    "1K": {"filter": None},
    "2K": {"filter": None}
}

for col_idx, (dataset, settings) in enumerate(datasets.items()):
    query = f'dataset == "{dataset}"'
    if settings["filter"]:
        query += f' and {settings["filter"]}'
    subset = tmp_df.query(query).copy()

    for row_idx, ratio in enumerate(['ratio_rep1', 'ratio_rep2', 'ratio_mean']):
        ax = axes[row_idx, col_idx]
        g = sns.histplot(data=subset, x=ratio, ax=ax, bins=100, element='step')
        g.axvline(x=subset[ratio].median(), ymin=0, ymax=200, color='black')

        if col_idx == 0:
            ylabel = {
                0: 'Ratio replicate 1',
                1: 'Ratio replicate 2',
                2: 'Ratio (mean)'
            }[row_idx]
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel(None)

        if row_idx == 2:
            ax.set_xlabel('Ratio')

plt.tight_layout()

if SAVE_FIGURES:
    plt.savefig(f'{FIGURE_DIR}/{SCRIPT_ID}-ratio_histograms.svg',**FIG_PARAMS)

mad_df = (
    tmp_df.query('ratio_rep2 < 10')
    .groupby('dataset')['ratio_mean']
    .apply(calculate_mad)
    .reset_index()
    .loc[[1, 3, 0, 2]]  # reorder datasets
    .rename(columns={'ratio_mean': 'mad'})
)
mad_dict = dict(zip(mad_df['dataset'], mad_df['mad']))

tmp_df = data_df.drop(columns=[
    'log_cov_rep1', 'log_cov_rep2', 'log_cov_rep3', 
    'protein_id', 'ncbi_id', 'tax_id'
]).copy()

fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(3, 4), sharex=True)

datasets = {
    "250": None,
    "500": "ratio_rep2 < 10",
    "1K": None,
    "2K": None
}

for ax, (dataset, filter_condition) in zip(axes, datasets.items()):
    query = f'dataset == "{dataset}"'
    if filter_condition:
        query += f' and {filter_condition}'
    subset = tmp_df.query(query).copy()

    median_val = subset['ratio_mean'].median()
    mad_val = mad_dict[dataset]

    g = sns.histplot(data=subset, x='ratio_mean', ax=ax, bins=100, element='step')
    g.axvline(x=median_val, ymin=0, ymax=200, color='black')
    g.axvline(x=median_val + mad_val, ymin=0, ymax=200, color='black', linestyle='--')
    g.axvline(x=median_val + mad_val * 2, ymin=0, ymax=200, color='black', linestyle='--')

    ax.set_ylabel(dataset)
    ax.set_xlabel('Ratio')

    ax.text(2.5, ax.get_ylim()[1] * 0.9, f"mad + median = {median_val + mad_val:.2f}",
            ha='center', va='top', fontsize=8)

plt.tight_layout()

if SAVE_FIGURES:
    plt.savefig(f'{FIGURE_DIR}/{SCRIPT_ID}-ratio_histograms_mad.svg',**FIG_PARAMS)