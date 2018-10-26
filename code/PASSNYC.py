# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 10:35:19 2018

@author: JiaRong
"""

# reference
# https://www.kaggle.com/infocusp/holographic-view-of-underperforming-schools/notebook

# Setting
%matplotlib inline

# import  libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter
import folium
import sklearn
import seaborn as sns
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
import sklearn
from sklearn.cluster import KMeans
import warnings 
import itertools
import base64
from IPython.display import HTML

warnings.filterwarnings('ignore')
init_notebook_mode(connected=True)

# Function converting string % values to int
def percent_to_int(df_in):
    for col in df_in.columns.values:
        if col.startswith("Percent") or col.endswith("%") or col.endswith("Rate"):
            df_in[col] = df_in[col].astype(np.object).str.replace('%','').astype(float)
    return df_in
        

df_schools_raw = pd.read_csv('../data/2016 School Explorer.csv')
df_schools_raw = df_schools_raw[df_schools_raw['Grade High']!='0K']
df_schools_raw = percent_to_int(df_schools_raw)
df_schools_raw['School Income Estimate'] = df_schools_raw['School Income Estimate'].astype(np.object).str.replace('$','').str.replace(',','').str.replace('.','').astype(float)

df_schools_relevant_grade = df_schools_raw[df_schools_raw['Grade High'].astype(int) > 5]

high_nan_columns = df_schools_raw.columns[df_schools_raw.isnull().mean() > 0.95]
# print("Here are the fields having >95% NaNs which we can drop: \n")
# print(list(high_nan_columns))

df_schools = df_schools_relevant_grade.drop(high_nan_columns, axis=1)
print("We have %d relevant schools and %d fields describing the school/ students"%(df_schools.shape))

def plot_city_hist(df_schools, title_str):
    layout = go.Layout(
            title = title_str,
            xaxis = dict(
                    title = 'City',
                    titlefont = dict(
                            family = 'Arial, sans-serif',
                            size = 12,
                            color = 'black'
                    ),
                    showticklabels = True,
                    tickangle = 315,
                    tickfont = dict(
                            size = 10,
                            color = 'grey'
                    )
            )
        )
    data = [go.Histogram(x = df_schools['City'])]
    fig = go.Figure(data = data, layout = layout)
    return fig
    
fig = plot_city_hist(df_schools, 'City wise School Distribution')
iplot(fig)























