import streamlit as st 
import numpy as np 

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

import pandas as pd
from pandas_datareader import data
aapl = data.DataReader("AAPL",
                        start='2015-1-1',
                        end='2019-12-31',
                        data_source='yahoo')

st.title('Streamlit Example')

st.write("""
# Different classifier and datasets
Compare classifiers:
""")
