

###---------- ASPECT DATA ANALYSIS

# import pandas and numpy packages
import pandas as pd
import numpy as np
# importing packages for the linear mixed effects model, ANOVA and Mann-Whitney U tests
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from scipy.stats import mannwhitneyu
from scipy import stats
# import seaborn for visualization
import seaborn as sns
from matplotlib import pyplot as plt

# definition of a function that helps us to calculate the
# avegerade and standard eviation per participant

def average_And_deviation(df, participand_id):
    
    # extracting only the part of the original data frame containing the
    # information about the pariticipant of interest
    local_df = df[df["Participant"]==participand_id]
    
    # create the list of columns we waant to look at
    list_of_columns = ["S1.RT","S2.RT","X5V","X4V","X3V",
                       "X2V","X1V","V","V1","V2","V3","V4",
                       "V5","S14.RT","S15.RT","S16.RT","S17.RT",
                       "S18.RT","S19.RT","S20.RT","S21.RT",
                       "S22.RT","S23.RT","S24.RT","S25.RT","S26.RT",
                       "S27.RT","S28.RT","S29.RT","S30.RT","S31.RT",
                       "S32.RT","S33.RT","S34.RT","S35.RT"]
    
    # creating empty list
    raw_list = []
    
    
    
    # the cutout (local_df) for our huge dataframe still has the same index as the original data frame
    # we have get the indices of the lines, which are in the cutout
    # my_index lists of indices for the cutout
    myIndex = local_df.index
    
    # outer for-loop: going through all the lines of the local data frame,
    # which contain only the information about one participant
    for i in myIndex:
    
        # inner for-loop: going through the columns of interest
        for j in (list_of_columns):
           
            #print ("column = "+ j )
            #print ("value + " + str(local_df[j][i]))
            raw_list.append(local_df[j][i])
     
    # now, raw_list contains all values of interest of the participant
        
    # calcuating the average and standard deviation, excluding NaN values
    average = np.nanmean(raw_list)    
    standard_deviation = np.nanstd(raw_list)
    
    print (raw_list)
    return average, standard_deviation
    
###---------- definition of average calulation completed


###---------- GERMAN GROUP ANALYSIS

#Reading the participant file in .csv format
df1 = pd.read_csv("TA_GER_ordered.csv", encoding = "ISO-8859-1", header=0, sep=",")

# creating 'Accuracy' column based on the correct responses and participants' answers
conditions = [
    (df1['Correct'] == 'x') & (df1['Response'] == 'x'),
    (df1['Correct'] == 'm') & (df1['Response'] == 'm')]
choices = [1, 1]
df1['Accuracy'] = np.select(conditions, choices, default=0)
print(df1)
# calculating the percentage of accurate responses
(df1.Accuracy.mean())*100
# calculating the percentage of inacurate responses
100-(df1.Accuracy.mean())*100
# saving the data frame with accuracy data before deleting inaccurate responses for analysis
Accuracy_G = df1
# dropping inacurate responses, because
# only the participants who paid attention during the experiment will be analyzed
df1 = df1[df1.Accuracy == 1]


# splitting the Conditions into factors for lmer
df1[['Tense', 'Type']] = df1.Condition.str.rsplit("_", n = 1, expand=True,)

# replacing 0 with missing data
# a zero would falsely indicate the speed of the response
# and it would affect the analysis of reaaction times data
df1.replace(0, np.nan, inplace=True)

# adding index for columns
idx = df1.columns 
label = df1.columns[0] 
lst = df1.columns.tolist()

# checking data types
d1types = df1.dtypes

# changing column types from integer to object
df1.Participant = df1.Participant.astype(object)


#----- calulating average (M) and standard deviation (SD) for each participant

# getting the participant numbers from the data frame
participant_list = df1["Participant"].unique()

# create new, empty columns
df1["average"] = 0
df1["StandardDeviation"] = 0


# iteration for all participants
for i in participant_list:
    
    
    # call the function and calculate M and SD for the participant with
    # the number stored in variable i
    my_result  = average_And_deviation(df1,i)
    current_M  = my_result[0]
    current_SD = my_result[1] 
    
    
    # get the indices belong to the current participant
    my_index = df1[df1["Participant"]==i].index

    # assinging to all lines of the M and SD column the value 
    # that the function gave us of the current participants
    df1.loc[my_index,"average"]           = current_M
    df1.loc[my_index,"StandardDeviation"] = current_SD

df1.to_excel("text_results_GER.xlsx")

#----- outlier correction
list_of_columns = ["S1.RT","S2.RT","X5V","X4V","X3V",
                       "X2V","X1V","V","V1","V2","V3","V4",
                       "V5","S14.RT","S15.RT","S16.RT","S17.RT",
                       "S18.RT","S19.RT","S20.RT","S21.RT",
                       "S22.RT","S23.RT","S24.RT","S25.RT","S26.RT",
                       "S27.RT","S28.RT","S29.RT","S30.RT","S31.RT",
                       "S32.RT","S33.RT","S34.RT","S35.RT"]
list_of_rows = df1.index

df1["outlier"] = 0

# iteration over all rows of the data frame

for i in list_of_rows:
    # iteration over all relevant columns
    for j in list_of_columns:
        
        # calculation of reference value: M + 2xSD
        reference_value = df1["average"][i] + 2*df1["StandardDeviation"][i]
        
        # if the specific value is larger than the reference, it is replaced by the reference
        if df1[j][i] > reference_value:
            df1.loc[i,j]           = reference_value
#----- outlier correction completed

## LMER
# Reading the Proficinecy file with 'LexTALE' and 'Cloze Test' columns in a .csv format
LexTALE = pd.read_csv("TA_LexTALE.csv", encoding = "ISO-8859-1", header=0, sep=",")

# Merging a file with segments with the proficinecy information
df1 = pd.merge(df1, LexTALE, on='Participant')
#checking data types
df1.info(verbose=True)
# changing column types
df1.Item = df1.Item.astype(object)

## getting ready for visualizing the data
# new file pnly with columns necessary for the graph
df1_graph = df1.filter(['Group', 'Condition', 'Participant', 'Item', 'V', 'V1', 'V2', 'V3','LexTALE', 'Tense', 'Type'])
# Sements melted into one column for plotting 
df1_graph = pd.melt(df1_graph, id_vars =['Group','Condition', 'Participant', 'Item','LexTALE', 'Tense', 'Type'], value_vars =['V', 'V1', 'V2', 'V3'])
# renaming the columns
df1_graph.rename(columns={'variable':'Segments', 'value': 'RT'}, inplace=True)
# line plot
Legend = sns.lineplot(y='RT', x='Segments', 
                 data=df1_graph, 
                 palette="colorblind",
                 hue='Condition', err_style=None,
                 hue_order=['Past_Match','Past_Mismatch','Perfect_Match','Perfect_Mismatch'])
Legend.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=False, ncol=1)
plt.show()

## LMER models for L1 German group
# divide the dataset into Past Simple and Present Perfect
PS1 = df1[df1.Tense == 'Past']
PP1 = df1[df1.Tense == 'Perfect']

# Past simple
# filtering columns
PS1 = PS1.filter(['Group', 'Condition', 'Participant', 'Item', 'Trial', 'V', 'V1', 'V2', 'V3','LexTALE', 'Tense', 'Type'])
# melting Segments into one column
PS1_graph = pd.melt(PS1, id_vars =['Group','Condition', 'Participant', 'Item', 'Trial','LexTALE', 'Tense', 'Type'], value_vars =['V', 'V1', 'V2', 'V3'])
# renaming columns
PS1_graph.rename(columns={'variable':'Segments', 'value': 'RT'}, inplace=True)

# line plot for Past Simple
Legend = sns.lineplot(y='RT', x='Segments', 
                 data=PS1_graph, 
                 palette="colorblind",
                 hue='Condition', err_style=None,
                 hue_order=['Past_Match','Past_Mismatch'])
Legend.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=False, ncol=1)
plt.show()

# LMER
# Region: V
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("V ~ Type*scale(LexTALE)", missing='drop', groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", data=PS1)
mdf = md.fit(method="nm")
print(mdf.summary())
# Region: V1
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("V1 ~ Type*scale(LexTALE)", missing='drop', groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", data=PS1)
mdf = md.fit(method="nm")
print(mdf.summary())
# Region: V2
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("V2 ~ Type*scale(LexTALE)", missing='drop', groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", data=PS1)
mdf = md.fit(method="nm")
print(mdf.summary())
# Region: V3
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("V3 ~ Type*scale(LexTALE)", missing='drop', groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", data=PS1)
mdf = md.fit(method="nm")
print(mdf.summary())

# Present perfect
# filtering columns
PP1 = PP1.filter(['Group', 'Condition', 'Participant', 'Item', 'Trial', 'V', 'V1', 'V2', 'V3','LexTALE', 'Tense', 'Type'])
# melting Segments into one column
PP1_graph = pd.melt(PP1, id_vars =['Group','Condition', 'Participant', 'Item', 'Trial','LexTALE', 'Tense', 'Type'], value_vars =['V', 'V1', 'V2', 'V3'])
# renaming columns
PP1_graph.rename(columns={'variable':'Segments', 'value': 'RT'}, inplace=True)

# line plot
Legend = sns.lineplot(y='RT', x='Segments', 
                 data=PP1_graph, 
                 palette="colorblind",
                 hue='Condition', err_style=None,
                 hue_order=['Perfect_Match','Perfect_Mismatch'])
Legend.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=False, ncol=1)
plt.show()

# LMER
# Region: V
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("V ~ Type*scale(LexTALE)", missing='drop', groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", data=PP1)
mdf = md.fit(method="nm")
print(mdf.summary())
# Region: V1
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("V1 ~ Type*scale(LexTALE)", missing='drop', groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", data=PP1)
mdf = md.fit(method="nm")
print(mdf.summary())
# Region: V2
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("V2 ~ Type*scale(LexTALE)", missing='drop', groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", data=PP1)
mdf = md.fit(method="nm")
print(mdf.summary())
# Region: V3
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("V3 ~ Type*scale(LexTALE)", missing='drop', groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", data=PP1)
mdf = md.fit(method="nm")
print(mdf.summary())


# LMER analysis: no separation of tenses
# Region: V
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("V ~ Tense*Type*scale(LexTALE)", missing='drop', groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", data=df1)
mdf = md.fit()
print(mdf.summary())
# Region: V1
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("V1 ~ Tense*Type*scale(LexTALE)", missing='drop', groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", data=df1)
mdf = md.fit()
print(mdf.summary())
# Region: V2
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("V2 ~ Tense*Type*scale(LexTALE)", missing='drop', groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", data=df1)
mdf = md.fit(method="nm")
print(mdf.summary())
# Region: V3
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("V3 ~ Tense*Type*scale(LexTALE)", missing='drop', groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", data=df1)
mdf = md.fit()
print(mdf.summary())


# creating a new file only with 6 areas or interest (AOI)
df1_only_segments = df1.filter(['Group', 'Participant', 'Item', 'LexTALE', 'Tense',
                      'Type', 'X1V', 'V', 'V1', 'V2', 'V3', 'V4', 'V5'])
# melt the segments columns into one column 
df1_all = pd.melt(df1_only_segments, id_vars =['Group', 'Participant', 'Item', 'LexTALE', 'Tense',
                      'Type'], 
                  value_vars =['X1V', 'V', 'V1', 'V2', 'V3', 'V4', 'V5'])
# rename columns into: 'Segments' and 'RT'
df1_all.rename(columns={'variable':'Segments', 'value': 'RT'}, inplace=True)

# Mixed model for all segments
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("RT ~ Segments + Tense*Type*scale(LexTALE)", missing='drop', groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", data=df1_all)
mdf = md.fit()
print(mdf.summary())
###---------- GERMAN GROUP ANALYSIS COMPLETED


###---------- CROATIAN GROUP ANALYSIS
# Reading the participant file in .csv format
df2 = pd.read_csv("TA_CRO_ordered.csv", encoding = "ISO-8859-1", header=0, sep=",")

# creating 'Accuracy' column based on the correct responses and participants' answers
conditions = [
    (df2['Correct'] == 'x') & (df2['Response'] == 'x'),
    (df2['Correct'] == 'm') & (df2['Response'] == 'm')]
choices = [1, 1]
df2['Accuracy'] = np.select(conditions, choices, default=0)
print(df2)
# calculating the percentage of accurate responses
(df2.Accuracy.mean())*100
# calculating the percentage of inaccurate responses
100-(df2.Accuracy.mean())*100
# saving the data frame with accuracy data before deleting inaccurate responses for analysis
Accuracy_C = df2
# dropping inacurate responses, because
# only the participants who paid attention during the experiment will be analyzed
df2 = df2[df2.Accuracy == 1]


# splitting the Conditions into factors for lmer
df2[['Tense', 'Type']] = df2.Condition.str.rsplit("_", n = 1, expand=True,)

# replacing 0 with missing data
# a zero would falsely indicate the speed of the response
# and it would affect the analysis of reaaction times data
df2.replace(0, np.nan, inplace=True)

# adding index for columns
idx = df2.columns 
label = df2.columns[0] 
lst = df2.columns.tolist()

# checking data types
d1types = df2.dtypes
# changing column types from integer to object
df2.Participant = df2.Participant.astype(object)

#----- calulating average (M) and standard deviation (SD) for each participant

# getting the participant numbers from the data frame
participant_list = df2["Participant"].unique()

# create new, empty columns
df2["average"] = 0
df2["StandardDeviation"] = 0


# iteration for all participants
for i in participant_list:
    
    
    # call the function and calculate M and SD for the participant
    # with the number stored in variable i
    my_result  = average_And_deviation(df2,i)
    current_M  = my_result[0]
    current_SD = my_result[1] 
    
    
    # get the index that belongs to the current participant
    my_index = df2[df2["Participant"]==i].index

    # assinging to all lines of the M and SD column the value 
    # that the function gave us of the current participants
    df2.loc[my_index,"average"]           = current_M
    df2.loc[my_index,"StandardDeviation"] = current_SD


#----- outlier correction
list_of_columns = ["S1.RT","S2.RT","X5V","X4V","X3V",
                       "X2V","X1V","V","V1","V2","V3","V4",
                       "V5","S14.RT","S15.RT","S16.RT","S17.RT",
                       "S18.RT","S19.RT","S20.RT","S21.RT",
                       "S22.RT","S23.RT","S24.RT","S25.RT","S26.RT",
                       "S27.RT","S28.RT","S29.RT","S30.RT","S31.RT",
                       "S32.RT","S33.RT","S34.RT","S35.RT"]
list_of_rows = df2.index

df2["outlier"] = 0

# iteration over all rows of the data frame
for i in list_of_rows:
    # iteration over all relevant columns
    for j in list_of_columns:
        
        # calculation of reference value: M + 2xSD
        reference_value = df2["average"][i] + 2*df2["StandardDeviation"][i]
        
        # if the specific value is larger than the reference, it is replaced by the reference
        if df2[j][i] > reference_value:
            df2.loc[i,j]           = reference_value
#----- outlier correction completed

# Reading the file with proficiency (LexTALE and Cloze Test) info in a .csv format
df2 = pd.merge(df2, LexTALE, on='Participant')
# renaming columns
df2.rename(columns={"Group_x": "Group"}, inplace=True)

# getting ready for visualization
# filtering the data
df2_graph = df2.filter(['Group', 'Condition', 'Participant', 'Item', 'Trial', 'V', 'V1', 'V2', 'V3','LexTALE', 'Tense', 'Type'])
# melting segments into one column
df2_graph = pd.melt(df2_graph, id_vars =['Group','Condition', 'Participant', 'Item', 'Trial','LexTALE', 'Tense', 'Type'], value_vars =['V', 'V1', 'V2', 'V3'])
# renaming columns
df2_graph.rename(columns={'variable':'Segments', 'value': 'RT'}, inplace=True)

# line plot
Legend = sns.lineplot(y='RT', x='Segments', 
                 data=df2_graph, 
                 palette="colorblind",
                 hue='Condition', err_style=None, 
                 hue_order=['Past_Match','Past_Mismatch','Perfect_Match','Perfect_Mismatch'])
Legend.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=False, ncol=1)
plt.show()

# divide the dataset into Past Simple and Present Perfect for the analysis
PS2 = df2[df2.Tense == 'Past']
PP2 = df2[df2.Tense == 'Perfect']

# Past simple
# filtering the data
PS2 = PS2.filter(['Group', 'Condition', 'Participant', 'Item', 'Trial', 'V', 'V1', 'V2', 'V3','LexTALE', 'Tense', 'Type'])
# melting segments into one column
PS2_graph = pd.melt(PS2, id_vars =['Group','Condition', 'Participant', 'Item', 'Trial','LexTALE', 'Tense', 'Type'], value_vars =['V', 'V1', 'V2', 'V3'])
# renaming columns
PS2_graph.rename(columns={'variable':'Segments', 'value': 'RT'}, inplace=True)
# line plot
Legend = sns.lineplot(y='RT', x='Segments', 
                 data=PS2_graph, 
                 palette="colorblind",
                 hue='Condition', err_style=None,
                 hue_order=['Past_Match','Past_Mismatch'])
Legend.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=False, ncol=1)
plt.show()

# LMER
# Region: V
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("V ~ Type*scale(LexTALE)", missing='drop', groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", data=PS2)
mdf = md.fit(method="nm")
print(mdf.summary())
# Region: V1
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("V1 ~ Type*scale(LexTALE)", missing='drop', groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", data=PS2)
mdf = md.fit(method="nm")
print(mdf.summary())
# Region: V2
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("V2 ~ Type*scale(LexTALE)", missing='drop', groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", data=PS2)
mdf = md.fit(method="nm")
print(mdf.summary())
# Region: V3
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("V3 ~ Type*scale(LexTALE)", missing='drop', groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", data=PS2)
mdf = md.fit(method="nm")
print(mdf.summary())


# Present perfect
# filtering the data
PP2 = PP2.filter(['Group', 'Condition', 'Participant', 'Item', 'Trial', 'V', 'V1', 'V2', 'V3','LexTALE', 'Tense', 'Type'])
# melting segments into one column
PP2_graph = pd.melt(PP2, id_vars =['Group','Condition', 'Participant', 'Item', 'Trial','LexTALE', 'Tense', 'Type'], value_vars =['V', 'V1', 'V2', 'V3'])
# renaming columns
PP2_graph.rename(columns={'variable':'Segments', 'value': 'RT'}, inplace=True)

# line plot
Legend = sns.lineplot(y='RT', x='Segments', 
                 data=PP2_graph, 
                 palette="colorblind",
                 hue='Condition', err_style=None,
                 hue_order=['Perfect_Match','Perfect_Mismatch'])
Legend.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=False, ncol=1)
plt.show()

# LMER
# Region: V
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("V ~ Type*scale(LexTALE)", missing='drop', groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", data=PP2)
mdf = md.fit(method="nm")
print(mdf.summary())
# Region: V1
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("V1 ~ Type*scale(LexTALE)", missing='drop', groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", data=PP2)
mdf = md.fit(method="nm")
print(mdf.summary())
# Region: V2
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("V2 ~ Type*scale(LexTALE)", missing='drop', groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", data=PP2)
mdf = md.fit(method="nm")
print(mdf.summary())
# Region: V3
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("V3 ~ Type*scale(LexTALE)", missing='drop', groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", data=PP2)
mdf = md.fit(method="nm")
print(mdf.summary())


# LMER analysis: no separation of tenses
# Region: V
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("V ~ Tense*Type*scale(LexTALE)", missing='drop', groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", data=df2)
mdf = md.fit(method="nm")
print(mdf.summary())
# Region: V1
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("V1 ~ Tense*Type*scale(LexTALE)", missing='drop', groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", data=df2)
mdf = md.fit()
print(mdf.summary())
# Region: V2
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("V2 ~ Tense*Type*scale(LexTALE)", missing='drop', groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", data=df2)
mdf = md.fit(method="nm")
print(mdf.summary())
# Region: V3
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("V3 ~ Tense*Type*scale(LexTALE)", missing='drop', groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", data=df2)
mdf = md.fit()
print(mdf.summary())

# creating a new file only with 6 areas or interest (AOI)
df2_only_segments = df2.filter(['Trial', 'Participant', 'Item', 'LexTALE', 'Tense',
                      'Type', 'X1V', 'V', 'V1', 'V2', 'V3', 'V4', 'V5'])
# melt the segments columns into one column 
df2_all = pd.melt(df2_only_segments, id_vars =['Trial', 'Participant', 'Item', 'LexTALE', 'Tense',
                      'Type'], 
                  value_vars =['X1V', 'V', 'V1', 'V2', 'V3', 'V4', 'V5'])
#  rename columns into: 'Segments' and 'RT'
df2_all.rename(columns={'variable':'Segments', 'value': 'RT'}, inplace=True)

# Mixed model for all segments
df2_all["group"] = 1
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("RT ~ Segments + Tense*Type*scale(LexTALE)", missing='drop', groups="group",                                                    
                                vc_formula=vcf, re_formula="0", data=df2_all)
mdf = md.fit()
print(mdf.summary())



###---------- SPANISH GROUP ANALYSIS
# Reading the participant file in .csv format
df3 = pd.read_csv("TA_ESP_ordered.csv", encoding = "ISO-8859-1", header=0, sep=",")


# creating 'Accuracy' column based on the correct responses and participants' answers
conditions = [
    (df3['Correct'] == 'x') & (df3['Response'] == 'x'),
    (df3['Correct'] == 'm') & (df3['Response'] == 'm')]
choices = [1, 1]
df3['Accuracy'] = np.select(conditions, choices, default=0)
print(df3)
# calculating the percentage of accurate responses
(df3.Accuracy.mean())*100
# calculating the percentage of inaccurate responses
100-(df3.Accuracy.mean())*100
# saving the data frame with accuracy data before deleting inaccurate responses for analysis
Accuracy_S = df3
# dropping inacurate responses, because
# only the participants who paid attention during the experiment will be analyzed
df3 = df3[df3.Accuracy == 1]

# splitting the Conditions into factors for lmer
df3[['Tense', 'Type']] = df3.Condition.str.rsplit("_", n = 1, expand=True,)

# replacing 0 with missing data
# a zero would falsely indicate the speed of the response
# and it would affect the analysis of reaaction times data
df3.replace(0, np.nan, inplace=True)

# adding index for columns
idx = df3.columns 
label = df3.columns[0] 
lst = df3.columns.tolist()

# checking data types
d3types = df3.dtypes
# changing column types from integer to object
df3.Participant = df3.Participant.astype(object)


#----- calulating average (M) and standard deviation (SD) for each participant

# getting the participant number from the data frame
participant_list = df3["Participant"].unique()

# create new, empty columns
df3["average"] = 0
df3["StandardDeviation"] = 0


# iteration for all participants
for i in participant_list:
    
    
    # call the function and calculate M and SD for the participant
    # with the number stored in variable i
    my_result  = average_And_deviation(df3,i)
    current_M  = my_result[0]
    current_SD = my_result[1] 
    
    
    # get the index belongs to the current participant
    my_index = df3[df3["Participant"]==i].index

    # assinging to all lines of the M and SD column the value 
    # that the function gave us of the current participants
    df3.loc[my_index,"average"]           = current_M
    df3.loc[my_index,"StandardDeviation"] = current_SD

#----- outlier correction
list_of_columns = ["S1.RT","S2.RT","X5V","X4V","X3V",
                       "X2V","X1V","V","V1","V2","V3","V4",
                       "V5","S14.RT","S15.RT","S16.RT","S17.RT",
                       "S18.RT","S19.RT","S20.RT","S21.RT",
                       "S22.RT","S23.RT","S24.RT","S25.RT","S26.RT",
                       "S27.RT","S28.RT","S29.RT","S30.RT","S31.RT",
                       "S32.RT","S33.RT","S34.RT","S35.RT"]
list_of_rows = df3.index

df3["outlier"] = 0

# iteration over all rows of the data frame
for i in list_of_rows:
    # iteration over all relevant columns
    for j in list_of_columns:
        
        # calculation of reference value: M + 2xSD
        reference_value = df3["average"][i] + 2*df3["StandardDeviation"][i]
        
        # if the specific value is larger than the reference, it is replaced by the reference
        if df3[j][i] > reference_value:
            df3.loc[i,j]           = reference_value
#----- outlier correction completed

# Reading the file with proficiency (LexTALE and Cloze Test) info in a .csv format
df3 = pd.merge(df3, LexTALE, on='Participant')
# renaming columns
df3.rename(columns={"Group_x": "Group"}, inplace=True)

# getting ready for visualization
# filtering columns
df3_graph = df3.filter(['Group', 'Condition', 'Participant', 'Item', 'Trial', 'V', 'V1', 'V2', 'V3','LexTALE', 'Tense', 'Type'])
# melting segments into one column
df3_graph = pd.melt(df3_graph, id_vars =['Group','Condition', 'Participant', 'Item', 'Trial','LexTALE', 'Tense', 'Type'], value_vars =['V', 'V1', 'V2', 'V3'])
# renaming columns int: 'Segment' and 'RT'
df3_graph.rename(columns={'variable':'Segments', 'value': 'RT'}, inplace=True)

# line plot
Legend = sns.lineplot(y='RT', x='Segments', 
                 data=df3_graph, 
                 palette="colorblind",
                 hue='Condition', err_style=None, 
                 hue_order=['Past_Match','Past_Mismatch','Perfect_Match','Perfect_Mismatch'])
Legend.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=False, ncol=1)
plt.show()

# divide the dataset into Past Simple and Present Perfect for the analysis
PS3 = df3[df3.Tense == 'Past']
PP3 = df3[df3.Tense == 'Perfect']

# Past simple
# filtering columns
PS3 = PS3.filter(['Group', 'Condition', 'Participant', 'Item', 'Trial', 'V', 'V1', 'V2', 'V3','LexTALE', 'Tense', 'Type'])
# meging segments into one column
PS3_graph = pd.melt(PS3, id_vars =['Group','Condition', 'Participant', 'Item', 'Trial','LexTALE', 'Tense', 'Type'], value_vars =['V', 'V1', 'V2', 'V3'])
# renaming columns int: 'Segment' and 'RT'
PS3_graph.rename(columns={'variable':'Segments', 'value': 'RT'}, inplace=True)

# line plot
Legend = sns.lineplot(y='RT', x='Segments', 
                 data=PS3_graph, 
                 palette="colorblind",
                 hue='Condition', err_style=None,
                 hue_order=['Past_Match','Past_Mismatch'])
Legend.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=False, ncol=1)
plt.show()

# LMER
# Region: V
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("V ~ Type*scale(LexTALE)", missing='drop', groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", data=PS3)
mdf = md.fit(method="nm")
print(mdf.summary())
# Region: V1
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("V1 ~ Type*scale(LexTALE)", missing='drop', groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", data=PS3)
mdf = md.fit(method="nm")
print(mdf.summary())
# Region: V2
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("V2 ~ Type*scale(LexTALE)", missing='drop', groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", data=PS3)
mdf = md.fit(method="nm")
print(mdf.summary())
# Region: V3
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("V3 ~ Type*scale(LexTALE)", missing='drop', groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", data=PS3)
mdf = md.fit(method="nm")
print(mdf.summary())

# Present perfect
# filter columns
PP3 = PP3.filter(['Group', 'Condition', 'Participant', 'Item', 'Trial', 'V', 'V1', 'V2', 'V3','LexTALE', 'Tense', 'Type'])
# melting segments into one column
PP3_graph = pd.melt(PP3, id_vars =['Group','Condition', 'Participant', 'Item', 'Trial','LexTALE', 'Tense', 'Type'], value_vars =['V', 'V1', 'V2', 'V3'])
# renaming columns int: 'Segment' and 'RT'
PP3_graph.rename(columns={'variable':'Segments', 'value': 'RT'}, inplace=True)

# ine plot
Legend = sns.lineplot(y='RT', x='Segments', 
                 data=PP3_graph, 
                 palette="colorblind",
                 hue='Condition', err_style=None,
                 hue_order=['Perfect_Match','Perfect_Mismatch'])
Legend.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=False, ncol=1)
plt.show()

# LMER
# Region: V
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("V ~ Type*scale(LexTALE)", missing='drop', groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", data=PP3)
mdf = md.fit(method="nm")
print(mdf.summary())
# Region: V1
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("V1 ~ Type*scale(LexTALE)", missing='drop', groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", data=PP3)
mdf = md.fit(method="nm")
print(mdf.summary())
# Region: V2
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("V2 ~ Type*scale(LexTALE)", missing='drop', groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", data=PP3)
mdf = md.fit(method="nm")
print(mdf.summary())
# Region: V3
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("V3 ~ Type*scale(LexTALE)", missing='drop', groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", data=PP3)
mdf = md.fit(method="nm")
print(mdf.summary())

# LMER analysis: no separation of tenses
# Region: V
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("V ~ Tense*Type*scale(LexTALE)", missing='drop', groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", data=df3)
mdf = md.fit()
print(mdf.summary())
# Region: V1
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("V1 ~ Tense*Type*scale(LexTALE)", missing='drop', groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", data=df3)
mdf = md.fit()
print(mdf.summary())
# Region: V2
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("V2 ~ Tense*Type*scale(LexTALE)", missing='drop', groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", data=df3)
mdf = md.fit()
print(mdf.summary())
# Region: V3
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("V3 ~ Tense*Type*scale(LexTALE)", missing='drop', groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", data=df3)
mdf = md.fit()
print(mdf.summary())


# creating a new file only with 6 areas or interest (AOI)
df3_only_segments = df3.filter(['Group', 'Participant', 'Item', 'LexTALE', 'Tense',
                      'Type', 'X1V', 'V', 'V1', 'V2', 'V3', 'V4', 'V5'])
# melf the segments into one column 
df3_all = pd.melt(df3_only_segments, id_vars =['Group', 'Participant', 'Item', 'LexTALE', 'Tense',
                      'Type'], 
                  value_vars =['X1V', 'V', 'V1', 'V2', 'V3', 'V4', 'V5'])
#  rename columns into: 'Segments' and 'RT'
df3_all.rename(columns={'variable':'Segments', 'value': 'RT'}, inplace=True)

# Mixed model for all segments
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("RT ~ Segments + Tense*Type*scale(LexTALE)", missing='drop', groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", data=df3_all)
mdf = md.fit()
print(mdf.summary())
###---------- SPANISH GROUP ANALYSIS COMPLETED

###---------- PLOTTING THE RESULTS OF ALL GROUPS TOGETHER
## Graph for Past Simple
graphPS = pd.read_csv("TA_Graph_PastSimple.csv", encoding = "ISO-8859-1", header=0, sep=",")
graph1 = pd.melt(graphPS, id_vars =['Group'], value_vars =['V', 'V+1', 'V+2','V+3'])
graph1.rename(columns={'variable':'Segments', 'value': 'RT'}, inplace=True)

# line plot
Legend = sns.lineplot(y='RT', x='Segments', 
                 data=graph1, 
                 palette="colorblind",
                 hue='Group', err_style=None, sort=False)
Legend.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=False, ncol=1)
plt.ylim(-140,140)
plt.show()

## Graph for Present Perfect
graphPP = pd.read_csv("TA_Graph_PresentPerfect.csv", encoding = "ISO-8859-1", header=0, sep=",")
graph2 = pd.melt(graphPP, id_vars =['Group'], value_vars =['V', 'V+1', 'V+2','V+3'])
graph2.rename(columns={'variable':'Segments', 'value': 'RT'}, inplace=True)

Legend = sns.lineplot(y='RT', x='Segments', 
                 data=graph2, 
                 palette="colorblind",
                 hue='Group', err_style=None, sort=False)
Legend.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=False, ncol=1)
plt.ylim(-140,140)
plt.show()
###---------- PLOTTING THE RESULTS COMPLETED


###---------- ANALYZING THE ACCURACY PERCENTAGE BETWEEN-GROUPS
# delete the unnecessary columns for all groups
Accuracy_G = Accuracy_G.filter(['Group', 'Accuracy', 'Participant', 'Item', 'LexTALE', 'Aspect',
                      'Type'])
Accuracy_C = Accuracy_C.filter(['Group', 'Accuracy', 'Participant', 'Item', 'LexTALE', 'Aspect',
                      'Type'])
Accuracy_S = Accuracy_S.filter(['Group', 'Accuracy', 'Participant', 'Item', 'LexTALE', 'Aspect',
                      'Type'])

# combine the three groups and their accuracies in one dataset
Accuracy = pd.concat([Accuracy_G, Accuracy_C,Accuracy_S], axis=0)
# all infromation (M, SD, min, max, etc.)
descriptives = Accuracy.groupby('Group').describe()
# ANOVA
mod = ols('Accuracy ~ Group', data=Accuracy).fit()
ACC_ANOVA = sm.stats.anova_lm(mod, typ=2)
print(ACC_ANOVA)
# testing normality
AccuracyNORM = stats.shapiro(Accuracy['Accuracy'])
AGC = mannwhitneyu(Accuracy_G['Accuracy'],Accuracy_C['Accuracy'])
AGS = mannwhitneyu(Accuracy_G['Accuracy'],Accuracy_S['Accuracy'])
ACS = mannwhitneyu(Accuracy_C['Accuracy'],Accuracy_S['Accuracy'])
###---------- ANALYZING ACCURACY COMPLETED


###---------- End of the analysis