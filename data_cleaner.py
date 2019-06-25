import pandas as pd
import numpy as np

import re
import string
import random
import operator

from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import torch
from torch.utils.data import Dataset, DataLoader

music = pd.read_csv('lyrics.csv')


# Initial clean of hip-hop dataframe
hip_hop_df = music[music.genre == "Hip-Hop"]

# 1. Take out all the rows with nan in lyrics
hip_hop_df = hip_hop_df[~hip_hop_df.lyrics.isna()]

# 2. Take out irregular years
num_yrs = len(sorted(hip_hop_df.year.unique()))
hip_hop_df = hip_hop_df[hip_hop_df.year >= 1989]

# Readjust index number of the dataframe
hip_hop_df = hip_hop_df.drop("index", axis=1)\
                       .reset_index()\
                       .drop("index", axis=1)

# Get number of "\n" for each lyrics
hip_hop_df["num_of_line_change"] = hip_hop_df.lyrics.apply(lambda x: x.count("\n"))

# Get the distribution of number of line changes
# Calculate the threshold of the filter
# The motivation is to take out
# 1. Intro
# 2. Outro
# 3. Interlude
# 4. Skit
thres_ = hip_hop_df.num_of_line_change.describe()["mean"] - hip_hop_df.num_of_line_change.describe()["std"]

# Filter out songs with number of line changes less than thres_
hip_hop_df = hip_hop_df[hip_hop_df.num_of_line_change >= int(thres_)]

# For consistency, we feel there is a need to set a maximum line length
# The motivation behind it is that some lyrics just have a 500 number of characters for a line

hip_hop_df["max_line_length"] = hip_hop_df["lyrics"].apply(lambda x: max([len(l) for l in x.split("\n")]))


# Calculate the threshold of the filter
# Get the data within 2 std's
lower_bound = hip_hop_df.max_line_length.describe()["mean"] - hip_hop_df.max_line_length.describe()["std"]
upper_bound = hip_hop_df.max_line_length.describe()["mean"] + hip_hop_df.max_line_length.describe()["std"]

# Filter out songs with number of max line length outside of 1 standard deviation
hip_hop_df = hip_hop_df[(hip_hop_df.max_line_length >= int(lower_bound)) &
                        (hip_hop_df.max_line_length <= int(upper_bound))]


# In the lyrics, there are many identifying lines that are
# actually not a part of the actual lyric in the song
# For example,
# - Chorus:
# - Verse 1:
# (Hook)
# We aim to take out these identifying lines

# Usually, these identifying lines are anotated with "[]" or "()"

# Build lists for bracket and parenthesis, respectively
bracket_list = []
parenthesis_list = []
i = 0 # processing flag

for lyrics in hip_hop_df.lyrics.values:
    # show process
    if i % 2000 == 0:
        print(i)
    lines = lyrics.split("\n")
    for line in lines:
        if line.startswith("[") or line.endswith("]") or line.startswith('{') or line.endswith('}'):
            bracket_list.append(line)
        if line.startswith("(") or line.endswith(")"):
            parenthesis_list.append(line)
    i += 1

bracket_dict = Counter(bracket_list)
parenthesis_dict = Counter(parenthesis_list)

# Get the filter for bracket
bracket_list_elim = []
for key in bracket_dict.keys():
    if (bracket_dict[key] >= 5 and len(key) <= 20) \
    or "verse" in key.lower() \
    or "chorus" in key.lower() \
    or key.endswith("}") \
    or key.endswith(")"):
        bracket_list_elim.append(key)

# Get the filter for parenthesis
parenthesis_list_elim = []
for key in parenthesis_dict.keys():
    if (parenthesis_dict[key] >= 5 and len(key) <= 15) \
    or "verse" in key.lower() \
    or "chorus" in key.lower():
        parenthesis_list_elim.append(key)

# Replace the parenthesis and bracket in the lyrics
def lyricFilter(lyric):
    lines = lyric.split("\n")
    temp_list = []
    for line in lines:
        if line not in bracket_list_elim and line not in parenthesis_list_elim and ":" not in line:
            temp_list.append(line)
    return '\n'.join(temp_list)

hip_hop_df_filtered = hip_hop_df.copy()
hip_hop_df_filtered["lyrics"] = hip_hop_df_filtered["lyrics"].apply(lambda x: lyricFilter(x))

hip_hop_df_filtered.to_csv("hip_hop_filtered.csv", index=False)
