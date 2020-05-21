# -----------------------------------------------------
#    Module d'initialisation  - Librairies a importer
# -----------------------------------------------------
import numpy                  as np
import pandas                 as pd
# Nombres avec sepa milliers "," et 3décimales après "."
pd.options.display.float_format = '{:,.3f}'.format   
# Option de transformation des "inf" en "na"
pd.options.mode.use_inf_as_na = True                

import seaborn                as sns; sns.set()
import matplotlib             as matplt
import matplotlib.pyplot      as plt
from   matplotlib             import patches
from   matplotlib.collections import LineCollection

import scipy                   as sc
import scipy.stats             as st
from scipy.stats               import t, shapiro
from   scipy.cluster.hierarchy import linkage, fcluster, dendrogram

import statsmodels.formula.api as smf
import statsmodels.api         as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from   sklearn                 import decomposition, preprocessing, cluster, metrics
from   sklearn.linear_model    import LinearRegression, LogisticRegression
from   sklearn.metrics         import confusion_matrix, roc_curve, auc
from   sklearn.model_selection import train_test_split

import six
from   math              import *

# /////  Pour gérer un affichage plus joli que la fonction "print"  //////
from IPython.display     import display, Markdown, HTML, display_html
def display_side_by_side(*args):
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)

# /////  Pour executer des requetes SQL de verification sur des DF  /////
from pandasql            import sqldf
execsql = lambda q: sqldf(q, globals())   
# EXEMPLE D'UTILISATION
# ----------------------
# req1 = ''' Select zone1, zone2 From DataFrame Where zone3=xx and zone4='xx' limit 3;'''
# df1 = execsql(req1)
# df1

import time

# Limiter le nombre de lignes d'affichage des Dataframes
pd.options.display.max_rows=20
