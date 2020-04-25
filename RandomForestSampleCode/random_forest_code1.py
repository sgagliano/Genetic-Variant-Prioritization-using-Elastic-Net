import sys, os
import numpy as np
import pylab as pl
from scipy import interp
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc, zero_one_score, precision_recall_curve, roc_auc_score
#from sklearn.ensemble import RandomForestClassifier
from forest import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold, permutation_test_score
from scipy.stats import fisher_exact, chi2_contingency, mannwhitneyu, scoreatpercentile
from pandas import *
from bisect import bisect
from scipy.stats.mstats import hmean
from math import log10

# read the RF_DIR from the environment, but default to the directory above where the script is located
RF_DIR = os.getenv('RF_DIR', '.')

#cols_to_drop    = ['labels.Chrom', 'labels.Pos']


def feat_imp(model, df, cls='cls'):
    idx = model.feature_importances_.argsort()   
    fis = Series(model.feature_importances_[idx[::-1]], index=df.columns.drop(cols_to_drop + [cls])[idx[::-1]].values)
    return fis

def build_full_classifier(df, cls='cls', rs=0):
    y = df[cls].values
    X = df.as_matrix(df.columns.drop(cols_to_drop + [cls]))
    classifier = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=6, random_state=rs, bootstrap=True, balance_classes=True, compute_importances=True, oob_score=True)
    model = classifier.fit(X, y)
    return model

def get_scores(df, model, cls='cls'):
    X = df.as_matrix(df.columns.drop(cols_to_drop + [cls]))
    res = model.predict_proba(X)
    return res[:,1]

def df_from_bed(fname, extra_cols=[], name_index=True):
    return read_table(fname, names=['cls'] + extra_cols, sep="\t", index_col=(1 if name_index else None))

def df_to_bed(df, fname, extra_cols=[], include_index=True, header=False, na_rep='', sort=True):
    bed_cols = ['cls']
    if not all([col in df for col in bed_cols]):
        raise Exception("DataFrame doesn't have necessary columns for BED format")
    if include_index:
        # include the index as the name field
        bed_cols.append('index')
    # remove any redundant columns from the extra cols list & add them to our list of columns
    cols = bed_cols + [e for e in extra_cols if e not in bed_cols]
    df.reset_index(inplace=True)
    if sort:
        df.sort(['cls'], inplace=True)
    df[cols].to_csv(fname, sep="\t", index=False, header=header, na_rep=na_rep)

def print_roc_auc(df, cls = 'cls'):
    y_true = df[cls].values
    y_scores = df['score'].values
    auc = roc_auc_score(y_true, y_scores)
    return auc
	
if __name__ == '__main__':

    annot_file = 'random_forest_test.csv'
    output_file = 'random_forest_result'

#    rd = RF_DIR+'/src/data_output/'

    hits = read_csv('random_forest_input_file1.csv', index_col=0) 

    model = build_full_classifier(hits)

    annot = read_csv(annot_file, index_col=0)

    annot['score'] = get_scores(annot, model)
    
    fis = feat_imp(model, annot)
    fis.to_csv(output_file+'_feat_imps.csv')
    
    df_to_bed(annot, output_file+'_scores.bed', extra_cols=['score'], header=True, sort=False)

    auc = print_roc_auc(annot)
    with open (output_file+'_roc.csv', "w") as text_file:
        text_file.write("ROC curve AUC: %s" % auc)
