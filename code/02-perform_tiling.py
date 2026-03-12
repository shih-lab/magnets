import json
import os
import pandas as pd
from Bio.Seq import Seq
from utils import fasta_to_dataframe, seqToTiles, protein_to_dna, codon_optimize
from globals import (
    TILE_LEN, TILE_SPACING, VIRUS_DIR, TILE_DIR,
    INFERENCE_DIR, LIBRARY_DIR, METADATA_DIR,
)

SCRIPT_ID = '02'
_config = json.loads(os.environ.get("SCRIPT_CONFIG", "{}"))

cluster_seqs = _config.get("cluster_seqs", False)
if cluster_seqs:
    os.system(f'cd-hit -i {VIRUS_DIR}/plant_viral_proteins.faa -o {TILE_DIR}/viral_proteins_clustered90.faa -c 0.9')

import_seqs = _config.get("import_seqs", False)
out_file = f'{TILE_DIR}/viral_proteins_clustered90.csv'
if import_seqs:
    protein_df = fasta_to_dataframe(f'{TILE_DIR}/viral_proteins_clustered90.faa')
    protein_df['genome'] = protein_df['description'].apply(lambda x: x.split(' | ')[1])
    protein_df['accession'] = protein_df['description'].apply(lambda x: x.split(' | ')[2])
    protein_df = protein_df.drop(columns='description').rename(columns={'sequence':'aa_seq','id':'protein_ID'})
    protein_df = protein_df.drop_duplicates(['protein_ID']).drop_duplicates('aa_seq').reset_index(drop=True)
    protein_df = protein_df.query('protein_ID != "None"')
    protein_df = protein_df[protein_df['aa_seq'].apply(lambda x: 'X' not in x and '*' not in x)] # 4537 -> 4364
    protein_df.to_csv(out_file,index=False)
else:
    protein_df = pd.read_csv(out_file)

make_tiles = _config.get("make_tiles", False)
out_file = f'{TILE_DIR}/viral_tiles.csv'
if make_tiles:
    tile_df = protein_df.copy()
    tile_df['tiles'] = tile_df['aa_seq'].apply(lambda x: seqToTiles(x, TILE_LEN, TILE_SPACING))
    tile_df = tile_df.explode('tiles',ignore_index=True)
    fill_len = len(str(tile_df['tiles'].apply(lambda x: x[0]).max()))
    tile_df['tile_count'] = tile_df['tiles'].apply(lambda x: str(x[0]).zfill(fill_len))
    tile_df['tile'] = tile_df['tiles'].apply(lambda x: x[1])
    tile_df['tile_ID'] = tile_df['protein_ID'] + '-' + tile_df['tile_count']
    tile_df = tile_df[tile_df['tile'].apply(lambda x: len(x) == TILE_LEN)].drop_duplicates('tile').reset_index(drop=True).drop(columns=['tiles','aa_seq'])
    tile_df.to_csv(out_file,index=False)
else:
    tile_df = pd.read_csv(out_file)

design_libraries = _config.get("design_libraries", False)
out_file = f'{LIBRARY_DIR}/viral_library.csv'
if design_libraries:
    predictions_df = pd.read_csv(f'{INFERENCE_DIR}/01-PADDLE/PADDLE_predictions.csv')
    merged_df = tile_df.merge(predictions_df.drop(columns='tile_ID'),on=['tile'])
    merged_df['zscore'] = merged_df['zscore'].astype(float)
    query_df = merged_df.nlargest(2000,'zscore').copy().reset_index(drop=True)
    metadata_df = pd.read_csv(f'{VIRUS_DIR}/plant_virus_metadata.csv')
    library_df = query_df.merge(metadata_df,left_on='genome',right_on='Filename')
    library_df = library_df.drop(columns=['Number of segments','Filename']).reset_index(drop=True).rename(columns={'activation':'PADDLE_activation','zscore':'PADDLE_zscore'})
    library_df['library_ID'] = [f'eLW044-{str(i+1).zfill(4)}' for i in library_df.index]
    codon_usage_df = pd.read_csv(f'{METADATA_DIR}/benthi_codon_usage.csv') # benthi codon usage from genscript
    aa_dict = {}
    for aa in codon_usage_df['Amino acid'].unique():
        codon_dict = {}
        for i, row in codon_usage_df.iterrows():
            if row['Amino acid'] == aa:
                codon_dict[row['Triplet']] = row['Fraction']
            else:
                continue
        fraction_sum = sum(codon_dict.values())
        aa_dict[aa] = codon_dict
    library_df['na_seq'] = library_df['tile'].apply(lambda x: codon_optimize(protein_to_dna(x),aa_dict))
    library_df = library_df.rename(columns={'tile':'aa_seq'})
    library_df['na_seq+overhangs'] = library_df['na_seq'].apply(lambda x: 'ACGTCAGtgtGGTCTCtaggt'.upper()+x.lower()+'TAAgcttagagaccaagcgctta'.upper())
    library_df = library_df[['library_ID','protein_ID','tile_ID','tile_count','aa_seq','na_seq','na_seq+overhangs','PADDLE_activation', 'PADDLE_zscore', 'genome', 'realm','accession',
        'RefSeq type', 'Source information', 'Genome length','Number of proteins', 'Genome Neighbors', 'Host', 'Date completed','Date updated']]
    assert all(library_df['aa_seq'] == library_df['na_seq'].apply(lambda x: Seq(x).translate()))
    assert all(library_df['na_seq'].apply(lambda x: 'GGTCTC' in x or 'CACCTGC' in x)) == False
    library_df.to_csv(out_file,index=False)
else:
    library_df = pd.read_csv(out_file)