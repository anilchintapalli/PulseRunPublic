import tensorflow as tf
import pandas as pd
import numpy as np
from scipy import stats


original_df = pd.read_csv('') #original data here
vae_df = pd.read_csv('') #synthetic data here

for column in original_df.columns:
    stat, p_value = stats.ks_2samp(original_df[column], vae_df[column])
    print(f"Feature: {column}, KS Statistic: {stat}, p-value: {p_value}")