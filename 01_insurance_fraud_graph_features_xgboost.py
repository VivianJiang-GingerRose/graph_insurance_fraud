# import package
from igraph import *
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from functools import reduce
import xgboost
import shap


# g = Graph()
# g.add_vertices(3)
# g.add_edges([(0,1), (1,2)])
#
# g.add_edges([(2, 0)])
# g.add_vertices(3)
# g.add_edges([(2, 3), (3, 4), (4, 5), (5, 3)])
# print(g)
# summary(g)


"""---------------------
Process in-patient data, including creating labels
------------------------"""
df_train_inp = pd.read_csv("data/Train_Inpatientdata-1542865627584.csv")
df_train_ben = pd.read_csv("data/Train_Beneficiarydata-1542865627584.csv")
df_train_fra = pd.read_csv("data/Train-1542865627584.csv")

df_train_inp['OperatingPhysician'].fillna('None', inplace=True)
df_train_inp['OtherPhysician'].fillna('None', inplace=True)

df_train_inp['ClaimStartDt'] = pd.to_datetime(df_train_inp['ClaimStartDt'], format = '%Y-%m-%d')
df_train_inp['ClaimEndDt'] = pd.to_datetime(df_train_inp['ClaimEndDt'], format = '%Y-%m-%d')

df_train_inp['ClaimDurationInDays'] = ((df_train_inp['ClaimEndDt']-df_train_inp['ClaimStartDt']).dt.days).astype('int64')

df_train_inp['AdmissionDt'] = pd.to_datetime(df_train_inp['AdmissionDt'], format='%Y-%m-%d')
df_train_inp['DischargeDt'] = pd.to_datetime(df_train_inp['DischargeDt'], format='%Y-%m-%d')
df_train_inp['AdmissionDurationInDays'] = ((df_train_inp['DischargeDt']-df_train_inp['AdmissionDt']).dt.days).astype('int64')
df_train_inp = df_train_inp.drop(columns=['ClaimEndDt', 'AdmissionDt', 'DischargeDt'])

# for ClmProcedureCodeCount in range(4):
#     df_train_inp['ClmProcedureCode_{}'.format(ClmProcedureCodeCount + 1)].fillna(0, inplace=True)
#
# for ClmDiagnosisCodeCount in range(10):
#     df_train_inp['ClmDiagnosisCode_{}'.format(ClmDiagnosisCodeCount + 1)].fillna('0', inplace=True)

diag_proce_col = ['ClmAdmitDiagnosisCode', 'ClmDiagnosisCode_1', 'ClmDiagnosisCode_10',
                  'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4',
                  'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7',
                  'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9', 'ClmProcedureCode_1',
                  'ClmProcedureCode_2', 'ClmProcedureCode_3', 'ClmProcedureCode_4',
                  'ClmProcedureCode_5', 'ClmProcedureCode_6']
df_train_inp[diag_proce_col] = df_train_inp[diag_proce_col].replace({np.nan:0})

df_train_inp.DeductibleAmtPaid.fillna(0, inplace=True)

"""Encode categorical columns"""
def encoded_cat(dataset, feature_to_encode='', col_list=[]):
    """This function returns top 5 cat column useful in determining potential fraud"""
    outer_list = []
    for col in col_list:
        list_1 = list()

        for item in list(dataset[col]):
            if str(item) == str(feature_to_encode):
                list_1.append(1)
            else:
                list_1.append(0)

        outer_list.append(list_1)

    li_sum = np.array([0] * 40474)

    for i in range(0, len(outer_list)):
        li1 = np.array(outer_list[i])
        li_sum = li_sum + li1

    return li_sum


procedure_col = ['ClmProcedureCode_1', 'ClmProcedureCode_2', 'ClmProcedureCode_3', 'ClmProcedureCode_4',
                 'ClmProcedureCode_5', 'ClmProcedureCode_6']

diagnosis_col = ['ClmDiagnosisCode_1', 'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4',
                 'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8',
                 'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10']

#top 5 procedure
# 4019.0, 2724.0, 9904.0, 8154.0, 66.0
df_train_inp['pr_4019'] = encoded_cat(df_train_inp, '4019.0', procedure_col)
df_train_inp['pr_2724'] = encoded_cat(df_train_inp, '2724.0', procedure_col)
df_train_inp['pr_9904'] = encoded_cat(df_train_inp,'9904.0',procedure_col)
df_train_inp['pr_8154'] = encoded_cat(df_train_inp,'8154.0',procedure_col)
df_train_inp['pr_66'] = encoded_cat(df_train_inp,'66.0',procedure_col)

#top 5 diagnosis 
#4019,25000,2724,V5869,42731
df_train_inp['di_4019'] = encoded_cat(df_train_inp, '4019' ,diagnosis_col)
df_train_inp['di_25000'] = encoded_cat(df_train_inp, '25000' ,diagnosis_col)
df_train_inp['di_2724'] = encoded_cat(df_train_inp, '2724' ,diagnosis_col)
df_train_inp['di_V5869'] = encoded_cat(df_train_inp,'V5869',diagnosis_col)
df_train_inp['di_42731'] = encoded_cat(df_train_inp,'42731',diagnosis_col)

for i in diag_proce_col:
    df_train_inp[i][df_train_inp[i] != 0] = 1

df_train_inp[diag_proce_col] = df_train_inp[diag_proce_col].astype(float)

# adding column total_num_diag
df_train_inp['total_num_diag'] = 0
for col in diagnosis_col:
    df_train_inp['total_num_diag'] = df_train_inp['total_num_diag'] + df_train_inp[col]

# adding column total_num_proce
df_train_inp['total_num_proce'] = 0
for col in procedure_col:
    df_train_inp['total_num_proce'] = df_train_inp['total_num_proce'] + df_train_inp[col]

df_train_inp['total_num_diag'] = df_train_inp['total_num_diag'].astype(float)
df_train_inp['total_num_proce'] = df_train_inp['total_num_proce'].astype(float)

## Merge inpatient data with benefits data
df_train_inp_m = pd.merge(df_train_inp, df_train_ben, on='BeneID', how='inner')

label_encoder = preprocessing.LabelEncoder()
df_train_fra['PotentialFraud'] = label_encoder.fit_transform(df_train_fra['PotentialFraud'])
df_train_inp_m = pd.merge(df_train_inp_m, df_train_fra, on='Provider', how='inner')

"""---------------------
Build graph features
------------------------"""
"""*********************
create graph from dataframe #1: ClaimID -> BenefitID
*********************"""
G1 = Graph.DataFrame(df_train_inp_m[['ClaimID', 'BeneID']], directed=False)
# t1 = df_train_inp_m[df_train_inp_m['BeneID'] == 'BENE100075']

## degree
degree1 = pd.DataFrame({'Node': G1.vs["name"], 'g_degree_BeneID': G1.strength()})
## closeness
closeness1 = pd.DataFrame({'Node': G1.vs["name"], 'g_closeness_BeneID': G1.closeness()})
## infomap
communities_infomap1 = pd.DataFrame({'Node': G1.vs["name"], 'g_communities_infomap_BeneID': G1.community_infomap().membership})

"""*********************
create graph from dataframe #2: ClaimID -> Provider
*********************"""
G2 = Graph.DataFrame(df_train_inp_m[['ClaimID', 'Provider']], directed=False)

## degree
degree2 = pd.DataFrame({'Node': G2.vs["name"], 'g_degree_Provider': G2.strength()})
## closeness
closeness2 = pd.DataFrame({'Node': G2.vs["name"], 'g_closeness_Provider': G2.closeness()})
## infomap
communities_infomap2 = pd.DataFrame({'Node': G2.vs["name"], 'g_communities_infomap_Provider': G2.community_infomap().membership})

"""*********************
create graph from dataframe #3: ClaimID -> Attending Physician
*********************"""
G3 = Graph.DataFrame(df_train_inp_m.loc[df_train_inp_m['AttendingPhysician'].notnull(),
                                        ['ClaimID', 'AttendingPhysician']], directed=False)

## degree
degree3 = pd.DataFrame({'Node': G3.vs["name"], 'g_degree_AttendingPhysician': G3.strength()})
## closeness
closeness3 = pd.DataFrame({'Node': G3.vs["name"], 'g_closeness_AttendingPhysician': G3.closeness()})
## infomap
communities_infomap3 = pd.DataFrame({'Node': G3.vs["name"], 'g_communities_infomap_AttendingPhysician': G3.community_infomap().membership})


# merge graph features
graph_feature = [degree1, closeness1, communities_infomap1, degree2, closeness2, communities_infomap2,
                 degree3, closeness3, communities_infomap3]
graph_feature = reduce(lambda left, right: pd.merge(left, right, how='left', on='Node'), graph_feature)

"""---------------------
Append graph features to a seperate train and test set
------------------------"""
df_train_inp_m1 = df_train_inp_m.set_index('ClaimID')

## One BeneID can have multiple claims, first extract the value
degree1_g = graph_feature.loc[graph_feature['Node'].str.contains('BENE'), ['Node', 'g_degree_BeneID']]
## Now get the BeneID to ClaimID lookup and create features at ClaimID level
degree1_g1 = pd.merge(df_train_inp_m[['ClaimID', 'BeneID']].drop_duplicates(), degree1_g,
                      how='inner', left_on='BeneID', right_on='Node').set_index('ClaimID')
## Append to training data
df_train_inp_m1 = pd.concat([df_train_inp_m1, degree1_g1[['g_degree_BeneID']]], axis=1)

## One provider can have multiple ClaimID, first extract the value
degree2_g = degree2.loc[degree2['Node'].str.contains('PRV'), ['Node', 'g_degree_Provider']]
## Now get the BeneID to ClaimID lookup and create features at ClaimID level
degree2_g1 = pd.merge(df_train_inp_m[['ClaimID', 'Provider']].drop_duplicates(), degree2_g,
                      how='inner', left_on='Provider', right_on='Node').set_index('ClaimID')
## Append to training data
df_train_inp_m1 = pd.concat([df_train_inp_m1, degree2_g1[['g_degree_Provider']]], axis=1)

## One Physician can have multiple ClaimID, first extract the value
degree3_g = degree3.loc[degree3['Node'].str.contains('PHY'), ['Node', 'g_degree_AttendingPhysician']]
## Now get the BeneID to ClaimID lookup and create features at ClaimID level
degree3_g1 = pd.merge(df_train_inp_m.loc[df_train_inp_m['AttendingPhysician'].notnull(), ['ClaimID', 'AttendingPhysician']].drop_duplicates(),
                      degree3_g, how='inner', left_on='AttendingPhysician', right_on='Node').set_index('ClaimID')
## Append to training data
df_train_inp_m1 = pd.concat([df_train_inp_m1, degree3_g1[['g_degree_AttendingPhysician']]], axis=1)

## All other features can be appended at ClaimID level
graph_feature_1 = graph_feature[graph_feature['Node'].str.contains('CLM')].drop(['g_degree_BeneID', 'g_degree_Provider', 'g_degree_AttendingPhysician'], axis=1).set_index('Node')
df_train_inp_m1 = pd.concat([df_train_inp_m1, graph_feature_1], axis=1)

"""---------------------
Setting up training and test sets without graph features
------------------------"""
df_train_inp_m2 = df_train_inp_m1.drop(['BeneID', 'Provider', 'AttendingPhysician', 'OperatingPhysician',
                                       'OtherPhysician', 'ClmAdmitDiagnosisCode', 'DOB', 'DOD', 'Gender',
                                       'Race', 'RenalDiseaseIndicator', 'State', 'County', 'DiagnosisGroupCode'], axis=1)
df_train_inp_m2 = df_train_inp_m2.drop(procedure_col, axis=1)
df_train_inp_m2 = df_train_inp_m2.drop(diagnosis_col, axis=1)


train = df_train_inp_m2[df_train_inp_m2['ClaimStartDt'] < '2009-10-01'].drop('ClaimStartDt', axis=1)
print(train.shape) #(31780, 43)
test = df_train_inp_m2[df_train_inp_m2['ClaimStartDt'] >= '2009-10-01'].drop('ClaimStartDt', axis=1)
print(test.shape) #(8694, 43)

## Set up two model training sets, one with graph features, one without graph features
x_train_g = train.drop(axis=1, columns=['PotentialFraud'])
x_train_ng = x_train_g.loc[:, ~x_train_g.columns.str.startswith('g_')]
y_train = train['PotentialFraud']
x_test_g = test.drop(axis=1, columns=['PotentialFraud'])
x_test_ng = x_test_g.loc[:, ~x_test_g.columns.str.startswith('g_')]
y_test = test['PotentialFraud']

"""---------------------
Train GBM model with and without graph features
------------------------"""
## 1) fit model with graph features
xgb_model_g = xgboost.sklearn.XGBRegressor(eval_metric='auc',
                                           objective='binary:logistic',
                                           learning_rate=0.2,
                                           n_estimators=10,
                                           scale_pos_weight=1,
                                           gamma=0.5,
                                           min_child_weight=3,
                                           subsample=0.9,
                                           colsample_bytree=0.9,
                                           max_depth=2)
xgb_model_g.fit(x_train_g, y_train, verbose=True)

y_pred_g = xgb_model_g.predict(x_test_g)
gini_g = 2 * roc_auc_score(y_test, y_pred_g) - 1
print("gini with graph feature is: ", gini_g.round(4))
## gini with graph feature is:  0.7921

explainer_g = shap.TreeExplainer(xgb_model_g)
shap_g = explainer_g.shap_values(x_train_g)
shap_g = pd.DataFrame(shap_g)
shap_g.columns = x_train_g.columns
df_shap_g = pd.DataFrame(shap_g.sum(axis=0, skipna=True).abs()).sort_values(by=[0], ascending=False)
df_shap_g.columns = ["shap_value"]
print("top 5 features by shapley value for model with graph features are:", df_shap_g.head())
# top 5 features by shapley value for model with graph features are:                                            shap_value
# g_degree_Provider                         4021.962646
# g_closeness_AttendingPhysician             456.506958
# g_communities_infomap_Provider             305.000275
# g_communities_infomap_AttendingPhysician    51.691402
# IPAnnualDeductibleAmt                        0.000000

## Suspicious, why is degree to provider the most contributing factor???

## 2) fit model without graph features
xgb_model_ng = xgboost.sklearn.XGBRegressor(eval_metric='auc',
                                            objective='binary:logistic',
                                            learning_rate=0.2,
                                            n_estimators=10,
                                            scale_pos_weight=1,
                                            gamma=0.5,
                                            min_child_weight=3,
                                            subsample=0.9,
                                            colsample_bytree=0.9,
                                            max_depth=2)
xgb_model_ng.fit(x_train_ng, y_train, verbose=True)

y_pred_ng = xgb_model_ng.predict(x_test_ng)
gini_ng = 2 * roc_auc_score(y_test, y_pred_ng) - 1
print("gini WITHOUT graph feature is: ", gini_ng.round(4))
## gini WITHOUT graph feature is:  0.0112

explainer_ng = shap.TreeExplainer(xgb_model_ng)
shap_ng = explainer_ng.shap_values(x_train_ng)
shap_ng = pd.DataFrame(shap_ng)
shap_ng.columns = x_train_ng.columns
df_shap_ng = pd.DataFrame(shap_ng.sum(axis=0, skipna=True).abs()).sort_values(by=[0], ascending=False)
df_shap_ng.columns = ["shap_value"]
print("top 5 features by shapley value for model WITHOUT graph features are:", df_shap_ng.head())
# top 5 features by shapley value for model WITHOUT graph features are:                           shap_value
# IPAnnualReimbursementAmt   50.216663
# InscClaimAmtReimbursed     36.257278
# OPAnnualDeductibleAmt      10.685817
# ClaimDurationInDays         5.702850
# AdmissionDurationInDays     2.708233

## Model performs much worse without the graph features....
## Further investigation is needed