#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 13:08:37 2020

@author: ibarlow
"""

from scipy import stats
import statsmodels.stats.multitest as smm
import pandas as pd

def paired_stats_tests(features,
                       y_classes,
                       distributionType='normal',
                       threshold=0.05,
                       control_group='DMSO'):
    """
    author @ibarlow

    Function for doing pairwise statistical test across multiple features
    and then posthoc correction for multiple comparisons. All data is
    treated as either normal or not normal, FUNCTION WOULD BE IMPROVED
    IF DISTRIBUTION TEST WAS INCLUDED TO SELECT APPROPRIATE STAT TEST

    Returns two dataframes with uncorrected and corrected p-values for each
    feature

    Parameters
    ----------
    features : dataframe of m x n where m are observations and n are features
        DESCRIPTION.
    y_classes : list of classes
        DESCRIPTION.
    distributionType : TYPE
        The default is normal.
    threshold : TYPE, optional
        critical values false discovery rate (Q), ie. the % of false positive
          that you are willing to accept; Default = 5% (0.05)
    control_group : string
        DESCRIPTION. The default in DMSO

    Returns
    -------
    pVals : Dataframe
        dataframe containing raw p values after the pairwise test and
        column y_class
    bhP_values : DataFrame
        multilevel index dataframe containing only the pValues
        that are significant after controlling for false discovery using the
        benjamini hochberg procedure, which ranks the raw p values,
        then calculates a threshold as (i/m)*Q,  where i=rank, m=total number
        of samples, Q=false discovery rate; p values < (i/m)*Q are considered
        significant. Uncorrected p-values are returned where the null
        hypothesis has been rejected. Contains column y_class

    group_classes: list
        list of the classes used for the pairwise statistical tests


    """
    if distributionType.lower() == 'normal':
        test = stats.ttest_ind
    else:
        test = stats.ranksums

    print ('Control group: {}'.format(control_group))

    featlist = list(features.columns)
    features['y_class'] = y_classes
    features_grouped = features.groupby('y_class')
    groups = list(features_grouped.groups.keys())

    pVals = []
    for item in groups:
        if control_group in item:
            continue
        else:
            # control = tuple(s if type(s)!=str else control_group for s in item)
            control = control_group
            _vals = pd.Series(dtype = object)
            for f in featlist:
                try:
                    _vals[f] = test(features_grouped.get_group(item)[f].values,
                                    features_grouped.get_group(control)[f].values,
                                    )[1]
                    _vals['y_class'] = item
                except Exception as error:
                    print ('error processing {}, {}, {}'.format(f, item, error))
            pVals.append(_vals.to_frame().transpose())
    pVals = pd.concat(pVals)
    pVals.reset_index(drop=True, inplace=True)

    # correct for multiple comparisons
    bhP_values = pd.DataFrame(columns = pVals.columns)
    for i,r in pVals.iterrows():
        _corrArray = smm.multipletests(r.drop(index=['y_class']).values,
                                       alpha = threshold,
                                       method = 'fdr_bh',
                                       is_sorted = False,
                                       returnsorted= False) #applied by row
        _corrArray = r.drop(index =['y_class'])[_corrArray[0]]
        _corrArray['y_class'] = r['y_class']
        bhP_values = bhP_values.append(_corrArray,
                                       ignore_index=True)

    # make new column for number of significantly different
    # bhP_values['sumSig'] = bhP_values.notna().sum(axis=1)
    group_classes = list(bhP_values['y_class'])
    bhP_values.drop(columns='y_class',
                    inplace=True)

    return pVals, bhP_values, group_classes
