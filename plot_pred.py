# %%
import os
os.environ['MPLCONFIGDIR'] = '/dev/shm/'

import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, norm, spearmanr
sns.set_theme(style="whitegrid")

# %%
pwd = "aps360-group-45/"
df = pd.read_csv(pwd + "predict_dls.csv")
ml_df = {}
ml_df['logs'] = df['logS']
ml_df['pred'] = df['prediction']
good_plot(ml_df, 'logs', 'pred', 'dls', [-8,1], pwd)

# %%
def good_plot(ml_df, arg1, arg2, ext, lims, pwd):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.axline((0,0),slope=1, linestyle="--", color='black')
    ax.axhline(0, linestyle="-", color='black')
    ax.axvline(0, linestyle="-", color='black')

    sns.scatterplot(data=ml_df, x=arg1, y=arg2)

    textstr='\n'.join((
        r'$\mathrm{MAD}=%.4f$' % (np.mean(abs(ml_df[arg1]-ml_df[arg2])), ),
        r'$\mathrm{AMAX}=%.4f$' % (max(abs(ml_df[arg1]-ml_df[arg2])), ),
        r'$\mathrm{SD}=%.4f$' % (np.std(ml_df[arg1]-ml_df[arg2]), ),
        r'$\mathrm{R}^2=%.4f$' % (r2_score(ml_df[arg1], ml_df[arg2]), ),
        #r'$\rho=%.4f$' % (spearmanr(ml_df[arg1], ml_df[arg2])[0])))
        r'$\rho=%.4f$' % (pearsonr(ml_df[arg1], ml_df[arg2])[0])))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.75, 0.25, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    plt.savefig(pwd + arg1 + '_' + arg2 + '_' + ext + '.jpg')
# %%
def calc_pearson(pred, truth):
    """
    Calculate a Pearson R**2 + confidence intervals for a set of predicted and true values
    :param pred: predicted values
    :param truth: true (experimental values)
    :return: Pearson R**2, lower and upper 95% confidence intervals
    """
    pearson_r_val = pearsonr(truth, pred)[0]
    lower, upper = pearson_confidence(pearson_r_val, len(pred))
    return [x ** 2 for x in [pearson_r_val, lower, upper]]

# %%

def pearson_confidence(r, num, interval=0.95):
    """
    Calculate upper and lower 95% CI for a Pearson r (not R**2)
    Inspired by https://stats.stackexchange.com/questions/18887
    :param r: Pearson's R
    :param num: number of data points
    :param interval: confidence interval (0-1.0)
    :return: lower bound, upper bound
    """
    stderr = 1.0 / math.sqrt(num - 3)
    z_score = norm.ppf(interval)
    delta = z_score * stderr
    lower = math.tanh(math.atanh(r) - delta)
    upper = math.tanh(math.atanh(r) + delta)
    return lower, upper
# %%
calc_pearson(ml_df['pred'], ml_df['logs'])
# %%
