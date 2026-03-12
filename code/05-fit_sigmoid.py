import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import fit_sigmoid
from globals import (
    EXPERIMENT_DIR, ANALYSIS_DIR,
    FIGURE_DIR, SAVE_FIGURES, FIG_PARAMS,
)

SCRIPT_ID = '05'
_config = json.loads(os.environ.get("SCRIPT_CONFIG", "{}"))

data_df = pd.read_csv(f'{EXPERIMENT_DIR}/experimental_data.csv')
gfp_df = pd.concat([
    pd.read_csv(f'{EXPERIMENT_DIR}/plate_reader/250_500_1k_internal_controls_fluo.csv'),
    pd.read_csv(f'{EXPERIMENT_DIR}/plate_reader/2k_internal_controls_fluo.csv')
]).reset_index(drop=True).drop(columns=['Unnamed: 0.1','Unnamed: 0','index']).rename(columns={'tileID':'tile_ID'})
merged_df = data_df.merge(gfp_df,on='tile_ID')
merged_df.head()

def plot_sigmoid_fit(merged_df, query, fig_name, drop_idx=None):
    tmp_df = merged_df.query(query).melt(
        id_vars=['tile_ID', 'lognormRFP_mean', 'lognormRFP_std'],
        value_vars=['ratio_rep1', 'ratio_rep2'],
        value_name='ratio'
    )
    tmp_df['variable'] = tmp_df['variable'].map({'ratio_rep1':'Replicate 1','ratio_rep2':'Replicate 2'})
    if drop_idx is not None:
        tmp_df = tmp_df.drop(drop_idx)

    x_fit, y_fit, y_lower, y_upper, x_intercept, residual_std, x0, y_midpoint, x_upper_bound = fit_sigmoid(
        x_data=tmp_df['ratio'].values.astype(float),
        y_data=tmp_df['lognormRFP_mean'].values.astype(float),
    )

    plt.figure(figsize=(2.5, 2.5))
    sns.lineplot(x=x_fit, y=y_fit, color='skyblue', linewidth=1)
    plt.fill_between(x_fit, y_lower, y_upper, color='skyblue', alpha=0.2)
    g = sns.scatterplot(
        data=tmp_df,
        x='ratio', y='lognormRFP_mean', s=15, palette=['black','tomato'], hue='variable', edgecolor=None,
    )
    group1 = tmp_df.query('variable == "Replicate 1"')
    group2 = tmp_df.query('variable == "Replicate 2"')
    plt.errorbar(group1['ratio'], group1['lognormRFP_mean'], yerr=group1['lognormRFP_std'], fmt='none', ecolor='black', alpha=0.8, linewidth=0.5)
    plt.errorbar(group2['ratio'], group2['lognormRFP_mean'], yerr=group2['lognormRFP_std'], fmt='none', ecolor='tomato', alpha=0.8, linewidth=0.5)
    plt.axhline(y=y_midpoint, color='grey', linestyle='--',linewidth=0.5)
    plt.axvline(x=x_upper_bound, color='grey', linestyle='--',linewidth=0.5)
    g.set(xlabel='Ratio', ylabel='logRFP')
    g.legend(frameon=False,handlelength=0.0)
    plt.tight_layout()

    if SAVE_FIGURES:
        plt.savefig(f'{FIGURE_DIR}/{SCRIPT_ID}-{fig_name}.svg',**FIG_PARAMS)

plot_sigmoid_fit(merged_df, 'dataset == "250" and iscontrol == "yes"', 'sigmoid_250')
plot_sigmoid_fit(merged_df, 'dataset == "500" and iscontrol == "yes"', 'sigmoid_500', drop_idx=44)
plot_sigmoid_fit(merged_df, 'dataset == "1K" and iscontrol == "yes"', 'sigmoid_1K')
plot_sigmoid_fit(merged_df, 'dataset == "2K" and iscontrol == "no"', 'sigmoid_2K')

output_fitted_data = _config.get("output_fitted_data", False)
out_file = f'{ANALYSIS_DIR}/experimental_data_fitted.csv'
if output_fitted_data:
    data_df['is_activator'] = False
    threshold_dict = {
        '250': 0.9750694705084277,
        '500': 1.0246787247457614,
        '1K': 0.9671268810104781,
        '2K': 1.0934882543970346
    }
    for dataset, threshold in threshold_dict.items():
        mask = (
            (data_df['dataset'] == dataset) &
            ((data_df['ratio_rep1'] > threshold) | (data_df['ratio_rep2'] > threshold))
        )
        data_df.loc[mask, 'is_activator'] = True
    data_df.to_csv(out_file,index=False)