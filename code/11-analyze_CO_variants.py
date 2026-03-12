import json
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr, spearmanr
from utils import create_protein_tiles, tile_to_residue_profile, parse_metapredict
from globals import (
    METADATA_DIR, CONSTANS_DIR, INFERENCE_DIR, EXPERIMENT_DIR,
    ANALYSIS_DIR, FIGURE_DIR, SAVE_FIGURES, FIG_PARAMS,
)

SCRIPT_ID = '11'
_config = json.loads(os.environ.get("SCRIPT_CONFIG", "{}"))

slice_seqs = _config.get("slice_seqs", False)
if slice_seqs:
    df = pd.read_csv(f'{METADATA_DIR}/CO_all_variants.csv')[['ID','sequence']]
    df['tiles_53aa'] = df['sequence'].apply(lambda x: create_protein_tiles(x,tile_size=53,overlap_size=52))
    df_exploded = df.explode('tiles_53aa')
    df_exploded['tile_ID'] = (
        df_exploded.groupby('ID').cumcount()
        .apply(lambda x: str(x).zfill(2))
    )
    df_exploded['tile_ID'] = df_exploded['ID'] + '-' + df_exploded['tile_ID']
    df_exploded[['tile_ID','tiles_53aa']].reset_index(drop=True).to_csv(f'{CONSTANS_DIR}/co_slices_53aa.csv',index=False)

    df['tiles_40aa'] = df['sequence'].apply(lambda x: create_protein_tiles(x,tile_size=40,overlap_size=39))
    df_exploded = df.explode('tiles_40aa')
    df_exploded['tile_ID'] = (
        df_exploded.groupby('ID').cumcount()
        .apply(lambda x: str(x).zfill(2))
    )
    df_exploded['tile_ID'] = df_exploded['ID'] + '-' + df_exploded['tile_ID']
    df_exploded[['tile_ID','tiles_40aa']].reset_index(drop=True).to_csv(f'{CONSTANS_DIR}/co_slices_40aa.csv',index=False)

make_fasta = _config.get("make_fasta", False)
if make_fasta:
    df = pd.read_pickle('../02-output/05-analysis/co_slices.pkl')
    with open('COvariants.faa','w') as out_file:
        for _,row in df.iterrows():
            out_file.write(f'>{row["ID"]}\n{row["sequence"]}\n')

run_metapredict = _config.get("run_metapredict", False)
if run_metapredict:
    os.system('metapredict-predict-disorder COvariants.faa -d cuda')

paddle_df = pd.read_pickle(f'{INFERENCE_DIR}/01-PADDLE/CO_variants_PADDLE.pkl')
paddle_df['PADDLE_zscore_scaled'] = MinMaxScaler().fit_transform(paddle_df['PADDLE_zscore'].to_numpy().reshape(-1,1))
paddle_df['protein_ID'] = paddle_df['tile_ID'].apply(lambda x: x.split('-')[0])
paddle_df['tile_num'] = paddle_df['tile_ID'].apply(lambda x: x.split('-')[1])
paddle_df = paddle_df.sort_values(['protein_ID', 'tile_num']).groupby(['protein_ID'])['PADDLE_zscore_scaled'].agg(list).reset_index()
paddle_df['PADDLE_zscore_scaled'] = paddle_df['PADDLE_zscore_scaled'].apply(lambda x: tile_to_residue_profile(x,tile_size=53,seq_len=69))
paddle_df['protein_ID'] = paddle_df['protein_ID'].apply(lambda x: x.replace('CO_','').replace('CO',''))
tada_df = pd.concat([
    pd.read_csv(f'{CONSTANS_DIR}/co_slices_40aa.csv'),
    pd.read_pickle(f'{INFERENCE_DIR}/02-TADA/COvariants_TADA_predictions.pkl')],axis=1)
tada_df['y_test_hat_scaled'] = MinMaxScaler().fit_transform(tada_df['y_test_hat'].to_numpy().reshape(-1,1))
tada_df['protein_ID'] = tada_df['tile_ID'].apply(lambda x: x.split('-')[0])
tada_df['tile_num'] = tada_df['tile_ID'].apply(lambda x: x.split('-')[1])
tada_df = tada_df.sort_values(['protein_ID', 'tile_num']).groupby(['protein_ID'])['y_test_hat_scaled'].agg(list).reset_index()
tada_df['y_test_hat_scaled'] = tada_df['y_test_hat_scaled'].apply(lambda x: tile_to_residue_profile(x,tile_size=40,seq_len=69))
tada_df['protein_ID'] = tada_df['protein_ID'].apply(lambda x: x.replace('CO_','').replace('CO',''))

metapredict_df = parse_metapredict(f'{CONSTANS_DIR}/COvariants_disorder_scores.csv')
metapredict_df.columns = ['protein_ID','disorder_predictions']
metapredict_df['protein_ID'] = metapredict_df['protein_ID'].apply(lambda x: x.replace('CO_','').replace('CO',''))

experiment_df = pd.read_csv(f'{EXPERIMENT_DIR}/CO_variants/CO_data.csv').drop(columns=['Unnamed: 0']).rename(columns={'Tile ID':'protein_ID','sequence':'AA_seq'})
experiment_df['protein_ID'] = experiment_df['protein_ID'].apply(lambda x: x.replace('CO_','').replace('CO',''))

merged_df = paddle_df.merge(tada_df,on='protein_ID').merge(metapredict_df,on='protein_ID').merge(experiment_df,on='protein_ID')
merged_df['PADDLE_mean'] =  merged_df['PADDLE_zscore_scaled'].apply(lambda x: np.mean(x))
merged_df['PADDLE_median'] = merged_df['PADDLE_zscore_scaled'].apply(lambda x: np.median(x))
merged_df['PADDLE_max'] = merged_df['PADDLE_zscore_scaled'].apply(lambda x: np.max(x))
merged_df['PADDLE_min'] = merged_df['PADDLE_zscore_scaled'].apply(lambda x: np.min(x))

merged_df['TADA_mean'] =  merged_df['y_test_hat_scaled'].apply(lambda x: np.mean(x))
merged_df['TADA_median'] = merged_df['y_test_hat_scaled'].apply(lambda x: np.median(x))
merged_df['TADA_max'] = merged_df['y_test_hat_scaled'].apply(lambda x: np.max(x))
merged_df['TADA_min'] = merged_df['y_test_hat_scaled'].apply(lambda x: np.min(x))

merged_df['metapredict_mean'] =  merged_df['disorder_predictions'].apply(lambda x: np.mean(x))
merged_df['metapredict_median'] = merged_df['disorder_predictions'].apply(lambda x: np.median(x))
merged_df['metapredict_max'] = merged_df['disorder_predictions'].apply(lambda x: np.max(x))
merged_df['metapredict_min'] = merged_df['disorder_predictions'].apply(lambda x: np.min(x))
merged_df.head()

tmp = merged_df[['protein_ID','AA_seq','ratio','coverage','PADDLE_mean','PADDLE_median','TADA_mean','TADA_median','metapredict_mean','metapredict_median']]
tmp.to_csv(f'{ANALYSIS_DIR}/CO_variant_data.csv',index=False)

fig, axes = plt.subplots(ncols=3,figsize=(6,2.5),sharey=True)
for _,row in merged_df.iterrows():
    axes[0].plot(row['PADDLE_zscore_scaled'], label='PADDLE',color='#a6c194')
    axes[0].set_xlabel('Tile position')
    axes[0].set_title('PADDLE')
    axes[0].set_ylabel('Predicted activity')

    axes[1].plot(row['y_test_hat_scaled'], label='TADA',color='#85d5e0')
    axes[1].set_xlabel('Tile position')
    axes[1].set_title('TADA')

    axes[2].plot(row['disorder_predictions'], label='Metapredict',color='steelblue')
    axes[2].set_xlabel('Tile position')
    axes[2].set_ylabel('Predicted disorder')
    axes[2].set_title('Metapredict')
plt.tight_layout()

y_vars = ['PADDLE_mean', 'PADDLE_median', 'PADDLE_max','TADA_mean', 'TADA_median', 'TADA_max',]

fig, axes = plt.subplots(ncols=3,nrows=2,figsize=(4,3),sharex=True,sharey=True)
params = dict(s=5,edgecolor='None')
for idx, y in enumerate(y_vars):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    if row == 0:
        color = '#a6c194'
    else:
        color = '#85d5e0'
    sns.scatterplot(data=merged_df, x='ratio', y=y, ax=ax, color=color,**params)

    x_vals = merged_df['ratio']
    y_vals = merged_df[y]
    pearson_corr, _ = pearsonr(x_vals, y_vals)
    spearman_corr, _ = spearmanr(x_vals, y_vals)

    ax.text(0.5, 0.3,
            f"r  = {pearson_corr:.2f}\nρ = {spearman_corr:.2f}",
            transform=ax.transAxes,
            verticalalignment='top',
            fontsize=8)

axes[0,0].set_title('Mean')
axes[0,1].set_title('Median')
axes[0,2].set_title('Max')

axes[0,0].set_ylabel('PADDLE')
axes[1,0].set_ylabel('TADA')
axes[1,0].set_xlabel('Ratio')
axes[1,1].set_xlabel('Ratio')
axes[1,2].set_xlabel('Ratio')

axes[0,0].axvline(x=1,ymin=0,ymax=1,color='black',linewidth=0.5,linestyle='--')
axes[0,1].axvline(x=1,ymin=0,ymax=1,color='black',linewidth=0.5,linestyle='--')
axes[0,2].axvline(x=1,ymin=0,ymax=1,color='black',linewidth=0.5,linestyle='--')
axes[1,0].axvline(x=1,ymin=0,ymax=1,color='black',linewidth=0.5,linestyle='--')
axes[1,1].axvline(x=1,ymin=0,ymax=1,color='black',linewidth=0.5,linestyle='--')
axes[1,2].axvline(x=1,ymin=0,ymax=1,color='black',linewidth=0.5,linestyle='--')
plt.tight_layout()

if SAVE_FIGURES:
    plt.savefig(f'{FIGURE_DIR}/{SCRIPT_ID}-ratio_vs_predictor.svg',**FIG_PARAMS)

columns_to_correlate = ['PADDLE_mean','PADDLE_median', 'PADDLE_max',
                        'TADA_mean', 'TADA_median','TADA_max']
results = []
for col in columns_to_correlate:
    valid = merged_df[['ratio', col]].dropna()
    x = valid['ratio']
    y = valid[col]

    pearson_r, _ = pearsonr(x, y)
    spearman_r, _ = spearmanr(x, y)

    results.append({
        'column': col,
        'pearson_r': pearson_r,
        'spearman_r': spearman_r})

correlation_df = pd.DataFrame(results).set_index('column')

plt.figure(figsize=(1.5,1.5))
g = sns.scatterplot(data=merged_df,x='TADA_median',y='PADDLE_median',s=7.5, edgecolor=None, color='cadetblue')
g.set(xlabel='TADA',ylabel='PADDLE')

pearson_corr, _ = pearsonr(merged_df['TADA_median'], merged_df['PADDLE_median'])
spearman_corr, _ = spearmanr(merged_df['TADA_median'], merged_df['PADDLE_median'])

g.text(-2.25,0.75,
        f"r  = {pearson_corr:.2f}\nρ = {spearman_corr:.2f}",
        transform=g.transAxes,
        fontsize=8);

if SAVE_FIGURES:
    plt.savefig(f'{FIGURE_DIR}/{SCRIPT_ID}-tada_vs_paddle.svg',**FIG_PARAMS)