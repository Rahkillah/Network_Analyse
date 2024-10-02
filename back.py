import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn import tree, metrics
from sklearn.model_selection import cross_val_score

# Set the page configuration
st.set_page_config(page_title="Trafic Analyse", page_icon="🔍")

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Settings
pd.set_option('display.max_columns', None)
np.set_printoptions(precision=3)
sns.set(style="darkgrid")


# Load train data
train = pd.read_csv("Train_data.csv")

# Descriptive statistics
train.describe()

# Attack Class Distribution
train['class'].value_counts()

# the test data
uploaded_file = st.file_uploader("Choose a CSV file for testing", type="csv")

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    st.write("Test Data Loaded Successfully")
    st.dataframe(test_data.head())