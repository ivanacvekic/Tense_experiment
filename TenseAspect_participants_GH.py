#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 19:30:16 2019

@author: ivanacvekic
"""
###---------- ASPECT PARTICIPANT INFORMATION ANALYSIS

## Importing pandas and numpy packages
import pandas as pd
# importing scipy locally for the Shapiro-Wilk Test
from scipy import stats
# importing statistical models (ANOVA, lmer & Mann-Whitney U)
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from scipy.stats import mannwhitneyu
# import matplotlib and seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# Reading the participant file in .csv format
df = pd.read_csv("TA participants.csv", header=0, sep=",")

# find the keys of the dataframe
keys_of_dataFrame = df.keys() # attributes

# checking the first five rows/lines of the dataframe
print(df.head())
# checking data types of all columns
df.dtypes
# changing 'Participant' column from integer to object
df.Participant = df.Participant.astype(object)

## Descriptives
# calculating descriptives for all columns
df.describe()
# calculating descriptives (M, SD, min, max, etc.) by experimental Group
descriptives = df.groupby('Group').describe()
# calculating only means for all columns per Group
mean = df.groupby('Group').mean()

## Significance testing
# Using ANOVA to test significane between groups for:
# LexTALE - proficiency test in English
mod = ols('LexTALE ~ Group', data=df).fit()
LextTALE_ANOVA = sm.stats.anova_lm(mod, typ=2)
print(LextTALE_ANOVA)
# CFT - category fluency task for proficiency in English
mod = ols('CFT ~ Group', data=df).fit()
CFT_ANOVA = sm.stats.anova_lm(mod, typ=2)
print(CFT_ANOVA)
# Age of the participants
mod = ols('Age ~ Group', data=df).fit()
Age_ANOVA = sm.stats.anova_lm(mod, typ=2)
print(Age_ANOVA)
# Age of Acquisition of participants
mod = ols('AoA ~ Group', data=df).fit()
AoA_ANOVA = sm.stats.anova_lm(mod, typ=2)
print(AoA_ANOVA)
# Years of learning English
mod = ols('Years ~ Group', data=df).fit()
Years_ANOVA = sm.stats.anova_lm(mod, typ=2)
print(Years_ANOVA)
# AJT - Acceptability judgement task 
# i.e., testing participants' knowledge of Present Perfect and Past Simple
mod = ols('AJT ~ Group', data=df).fit()
AJT_ANOVA = sm.stats.anova_lm(mod, typ=2)
print(AJT_ANOVA)

# Renaming columns
# AJT 100 -> AJTP and Cloze test 100 into Cloze test P
df.rename(columns={"AJT 100": "AJTP", "Cloze test 100": "ClozeP", "Cloze test w PS": "ClozewPS", "Cloze test": "ClozeTest"}, inplace=True)

# AJT in percentage
mod = ols('AJTP ~ Group', data=df).fit()
AJTP_ANOVA = sm.stats.anova_lm(mod, typ=2)
print(AJTP_ANOVA)
# Cloze test - testing participants' knowledge of Past Simple and Present Perfect
mod = ols('ClozeTest ~ Group', data=df).fit()
Clozetest_ANOVA = sm.stats.anova_lm(mod, typ=2)
print(Clozetest_ANOVA)
# Cloze test for tense and aspect in percentage
mod = ols('ClozeP ~ Group', data=df).fit()
ClozeP_ANOVA = sm.stats.anova_lm(mod, typ=2)
print(ClozeP_ANOVA)
# Cloze test without Present Simple
mod = ols('ClozewPS ~ Group', data=df).fit()
ClozetestwPS_ANOVA = sm.stats.anova_lm(mod, typ=2)
print(ClozetestwPS_ANOVA)
# Cloze test without Present Simple in %
mod = ols('ClozePS100 ~ Group', data=df).fit()
ClozetestwPS_ANOVA = sm.stats.anova_lm(mod, typ=2)
print(ClozetestwPS_ANOVA)


## Testing normality of the data for significant results
# using the Shapiro–Wilk test for normal distribution testing
LexTALE_N = stats.shapiro(df['LexTALE'])
Age_N = stats.shapiro(df['Age'])
AoA_N = stats.shapiro(df['AoA'])
ClozeTest_N = stats.shapiro(df['ClozeTest'])
ClozeP_N = stats.shapiro(df['ClozeP'])
ClozewPS_N = stats.shapiro(df['ClozewPS'])
ClozewPSP_N = stats.shapiro(df['ClozePS100'])

# Creating subsets by Group for significance testing between-groups
S = df[df.Group == 'Spanish']
C = df[df.Group == 'Croatian']
G = df[df.Group == 'German']

## Significance testing between each Group pair
# in order to find out where the difference is

# German - Croatian
GC1 = mannwhitneyu(G['LexTALE'],C['LexTALE'])
GC2 = mannwhitneyu(G['Age'],C['Age'])
GC3 = mannwhitneyu(G['AoA'],C['AoA'])
GC4 = mannwhitneyu(G['ClozeTest'],C['ClozeTest'])
GC5 = mannwhitneyu(G['ClozeP'],C['ClozeP'])
GC6 = mannwhitneyu(G['ClozewPS'],C['ClozewPS'])
GC7 = mannwhitneyu(G['ClozePS100'],C['ClozePS100'])

# German - Spanish
GS1 = mannwhitneyu(G['LexTALE'],S['LexTALE'])
GS2 = mannwhitneyu(G['Age'],S['Age'])
GS3 = mannwhitneyu(G['AoA'],S['AoA'])
GS4 = mannwhitneyu(G['ClozeTest'],S['ClozeTest'])
GS5 = mannwhitneyu(G['ClozeP'],S['ClozeP'])
GS6 = mannwhitneyu(G['ClozewPS'],S['ClozewPS'])
GS7 = mannwhitneyu(G['ClozePS100'],S['ClozePS100'])

# Croatian - Spanish
CS1 = mannwhitneyu(C['LexTALE'],S['LexTALE'])
CS2 = mannwhitneyu(C['Age'],S['Age'])
CS3 = mannwhitneyu(C['AoA'],S['AoA'])
CS4 = mannwhitneyu(C['ClozeTest'],S['ClozeTest'])
CS5 = mannwhitneyu(C['ClozeP'],S['ClozeP'])
CS6 = mannwhitneyu(C['ClozewPS'],S['ClozewPS'])
CS7 = mannwhitneyu(C['ClozePS100'],S['ClozePS100'])

# Repeated measures ANOVA for AJT
# Make a new dataset with conditions only
MMM = df.filter(['Group', 'Participant', 'PS M', 'PS MM', 'PP M', 'PP MM'])
# renaming columns for analysis
MMM.rename(columns={"PS M": "Past Simple Match", "PS MM": "Past Simple Mismatch",
                    "PP M": "Present Perfect Match", "PP MM": "Present Perfect Mismatch"}, inplace=True)
# melt the columns into one 
MMMM = pd.melt(MMM, id_vars =['Group', 'Participant'], value_vars =['Past Simple Match', 'Past Simple Mismatch', 'Present Perfect Match', 'Present Perfect Mismatch'])
# rename the columns
MMMM.rename(columns={'variable':'Condition', 'value': 'Score'}, inplace=True)
# splitting the Conditions into factors
MMMM[['Tense', 'Type']] = MMMM.Condition.str.rsplit(" ", n = 1, expand=True,)

# separate groups for analysis
SMMM = MMMM[MMMM.Group == 'Spanish']
CMMM = MMMM[MMMM.Group == 'Croatian']
GMMM = MMMM[MMMM.Group == 'German']

## getting ready for visualizing the AJT data

# L1 German
# checking means of each condition
GMMM.groupby('Condition').describe()
# graph
sns.barplot(y='Condition', x='Score', data=GMMM, palette="colorblind", linewidth=2)

# L1 Croatian
# checking means of each condition
CMMM.groupby('Condition').describe()
# graph
sns.barplot(y='Condition', x='Score', data=CMMM, palette="colorblind", linewidth=2)

# L1 Spanish
# checking means of each condition
SMMM.groupby('Condition').describe()
# graph
sns.barplot(y='Condition', x='Score', data=SMMM, palette="colorblind", linewidth=2)

# Plotting all Groups in one graph
Legend = sns.barplot(y='Score', x='Group', 
                 data=MMMM, 
                 palette="colorblind",
                 hue='Condition')
Legend.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)
plt.show()

# Model for L1 Spanish
md = smf.mixedlm("Score ~ Tense*Type", SMMM, groups=SMMM["Group"])
mdf1 = md.fit(method="cg")
print(mdf1.summary())
# Model for L1 Croatian
md = smf.mixedlm("Score ~ Tense*Type", CMMM, groups=CMMM["Group"])
mdf2 = md.fit(method='nm')
print(mdf2.summary())
# Model for L1 German
md = smf.mixedlm("Score ~ Tense*Type", GMMM, groups=GMMM["Group"])
mdf3 = md.fit(method='nm')
print(mdf3.summary())
# Model for ALL
md = smf.mixedlm("Score ~ Tense*Type*Group", MMMM, groups=MMMM["Group"])
mdf = md.fit(method='cg')
print(mdf.summary())

#save in a fast format
df.to_pickle("TA.pkl")

###---------- End of the analysis