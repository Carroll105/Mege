# encoding=UTF-8
from __future__ import division
import numpy as np
import pandas as pd
from minepy import MINE
from sklearn.model_selection import LeaveOneOut
import scipy.stats as stats
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import chi2
from elm import ELMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import sys
stdsc = StandardScaler()

def load_data(filename, labelname):
    data = pd.read_table(filename, index_col=0)
    label = pd.read_table(labelname, index_col=0).values.ravel()
    return data, label

def classfiers(X, Y,times,fold):
    #用6个分类器对X，Y做交叉验证 times次数 fold折
    classifier = [SVC(kernel='linear'), GaussianNB(), KNeighborsClassifier(), DecisionTreeClassifier(random_state=10),
                  GradientBoostingClassifier(random_state=10), ELMClassifier()]
    acc_res = []
    for clf in classifier:
        if clf == classifier[-1]:
            X = stdsc.fit_transform(X)
        each_score = []
        for i in range(times):
            acc_temp = []
            skf = StratifiedKFold(n_splits=fold, random_state=i, shuffle=True)
            for train_index, test_index in skf.split(X, Y):
                # print('Train: ',train_index,'Test: ',test_index)
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]
                clf.fit(X_train, Y_train)
                acc1 = accuracy_score(Y_test, clf.predict(X_test))
                acc_temp.append(acc1)
            each_score.append(np.mean(acc_temp))
        acc_res.append(np.mean(each_score))
    return acc_res

def IFS_validation(X, Y, times,fold,select,tag):
    #对X进行IFS,select标志哪种特征选择算法,用于写文件名称
    feature_order = X.columns
    list_gene = []
    result = []
    print("feature_number:")
    for i in range(1, len(feature_order) + 1):
        print(i)
        list_gene.append(feature_order[:i])
        result.append(classfiers(X[feature_order[:i]].values, Y,times,fold))
    score = pd.DataFrame(result)
    col = ['SVM', 'GaussianNB', 'KNN', 'DTree', 'GBDT', 'ELM']
    score.columns = col
    score.index = list(map(lambda x: ','.join(x), list_gene))
    score.to_csv(tag + "_" + select + "_Acc_" + str(times) + "_runs_" + str(fold) + "_fold.csv")

def TRank(data, label, ordername,times,fold,cnt,tag):
    select = "TRank"
    X = data
    result = pd.read_table(ordername, index_col=0)
    result = result.iloc[list(range(cnt))]
    # print(result)
    print(select + " IFS validation start...")
    cur_X = X.loc[result.index]
    cur_X.to_csv(tag + "_" + select + "_" + str(cnt) +"_result.csv")
    cur_X = cur_X.T  # 纵轴为标签
    # print(cur_X)
    IFS_validation(cur_X, label,times,fold, select,tag)
    print(select + ' IFS validation finish...')
    print("------------------------")

def WRank(data, label,times,fold,cnt,tag,geneOrSite):
    select = "WRank"
    X = data
    print(select + ' selection start...')
    p_index = np.argwhere(label == 1).ravel()
    n_index = np.argwhere(label == 0).ravel()
    X.columns = list(range(len(label)))
    p_data = X[p_index]
    n_data = X[n_index]
    res = []
    for gene in p_data.index:
        data1 = p_data.loc[gene]
        data2 = n_data.loc[gene]
        res.append(stats.ranksums(data1, data2).pvalue)
    result = pd.DataFrame({geneOrSite: p_data.index, 'pvalue': res}).sort_values(by='pvalue')
    print(select + " result.head()")
    print(result.head())
    col = [geneOrSite, 'pvalue']
    result.to_csv(tag + '_WRank_rank.txt', index=None, columns=col, sep="\t")
    print(select + ' selection finish...')
    print(select + " IFS validation start...")
    result = result.iloc[list(range(cnt))]
    cur_X = X.loc[result[geneOrSite]]
    cur_X.to_csv(tag + "_" + select + "_" + str(cnt) + "_result.csv")
    cur_X = cur_X.T  # 纵轴为标签
    # print(cur_X)
    IFS_validation(cur_X, label,times,fold, select,tag)
    print(select + ' IFS validation finish...')
    print("------------------------")

def Chi2(data, label,times,fold,cnt,tag,geneOrSite):
    select = "Chi2"
    print('Chi2 selection start...')
    X = data.T  # 纵轴为标签
    # (0,1)化
    minMaxScaler = preprocessing.MinMaxScaler()
    minMax = minMaxScaler.fit_transform(X)
    X = pd.DataFrame(minMax, columns=X.columns.values)
    print('after standardized...')
    print(X.head())
    (chi, pval) = chi2(X, label)
    res = pd.DataFrame({geneOrSite: data.index.tolist(), 'chi2': chi}).sort_values(by='chi2', ascending=False)
    print('Chi2 result.head()')
    print(res.head())
    col = [geneOrSite, 'chi2']
    res.to_csv(tag + '_Chi2_rank.txt', sep="\t", index=None, columns=col)
    print('Chi2 selection finish...')
    print(select + " IFS validation start...")
    result = res.iloc[list(range(cnt))]
    cur_X = data.loc[result[geneOrSite]]
    cur_X.to_csv(tag + "_" + select + "_" + str(cnt) + "_result.csv")
    cur_X = cur_X.T  # 纵轴为标签
    IFS_validation(cur_X, label, times,fold,select,tag)
    print(select + ' IFS validation finish...')
    print("------------------------")

def ROCRank(data, label,times,fold,cnt,tag,geneOrSite):
    select = "ROCRank"
    print('ROCRank start...')
    X = data.T  # 纵轴为标签
    feat_labels = X.columns.tolist()
    auc_list = []
    clf = GaussianNB()
    random_seed = 0
    for item in feat_labels:
        X_train, X_test, y_train, y_test = train_test_split(X[item].values, label, test_size=0.3,
                                                            random_state=random_seed)
        clf.fit(X_train.reshape(-1, 1), y_train)
        y_pred = clf.predict_proba(X_test.reshape(-1, 1))
        auc_list.append(roc_auc_score(y_test, y_pred[:, 1]))
    result = pd.DataFrame({geneOrSite: feat_labels, 'AUC': auc_list}).sort_values(by='AUC', ascending=False)
    print('ROCRank result.head()')
    print(result.head())
    col = [geneOrSite, 'AUC']
    result.to_csv(tag + '_ROCRank_rank.txt', sep="\t", index=None, columns=col)
    print('ROCRank selection finish')
    print(select + " IFS validation start...")
    result = result.iloc[list(range(cnt))]
    cur_X = data.loc[result[geneOrSite]]
    cur_X.to_csv(tag + "_" + select + "_" + str(cnt) + "_result.csv")
    cur_X = cur_X.T  # 纵轴为标签
    IFS_validation(cur_X, label,times,fold, select,tag)
    print(select + ' IFS validation finish...')
    print("------------------------")

def RF(data, label,times,fold,cnt,tag,geneOrSite):
    select = "RF"
    print('RF selection start...')
    X = data.T  # 纵轴为标签
    forest = RandomForestClassifier(n_estimators=1000, random_state=10, n_jobs=1)
    forest.fit(X, label)
    importance = forest.feature_importances_
    result = pd.DataFrame({geneOrSite: X.columns.values, "importance": importance})
    result = result.sort_values(by='importance', ascending=False)
    print('RF result.head()')
    print(result.head())
    col = [geneOrSite, "importance"]
    result.to_csv(tag + "_RF_rank.txt", sep="\t", index=None, columns=col)
    print('RF selection finish...')
    print(select + " IFS validation start...")
    result = result.iloc[list(range(cnt))]
    cur_X = data.loc[result[geneOrSite]]
    cur_X.to_csv(tag + "_" + select + "_" + str(cnt) + "_result.csv")
    cur_X = cur_X.T  # 纵轴为标签
    IFS_validation(cur_X, label,times,fold, select,tag)
    print(select + ' IFS validation finish...')
    print("------------------------")

def SVM_RFE(data, label,times,fold,cnt,tag,geneOrSite,pvaluename):
    select = "SVM_RFE"
    print("SVM_RFE selection start...")
    if geneOrSite != "gene":  # 位点就用前2万个
        order = pd.read_table(pvaluename, index_col=0).iloc[list(range(20000))]
        # print(order)
        X = data.loc[order.index]
        X = X.T
    else:
        X = data.T  # 纵轴为标签
    # use svm as the model
    clf = SVC(kernel='linear', C=1)
    rfe = RFE(clf, n_features_to_select=1)
    rfe.fit(X, label)
    result = pd.DataFrame({geneOrSite: X.columns.values, "rank": rfe.ranking_}).sort_values(by='rank')
    print("SVM_RFE result.head()")
    print(result.head())
    col = [geneOrSite, 'rank']
    result.to_csv(tag + '_SVM_RFE_rank.txt', sep="\t", index=None, columns=col)
    print("SVM_RFE selection finish...")
    print(select + " IFS validation start...")
    result = result.iloc[list(range(cnt))]
    cur_X = data.loc[result[geneOrSite]]
    cur_X.to_csv(tag + "_" + select + "_" + str(cnt) + "_result.csv")
    cur_X = cur_X.T  # 纵轴为标签
    IFS_validation(cur_X, label,times,fold, select,tag)
    print(select + ' IFS validation finish...')
    print("------------------------")

def LR_RFE(data, label,times,fold,cnt,tag,geneOrSite,pvaluename):
    select = "LR_RFE"
    print(select + " selection start...")
    if geneOrSite != "gene":
        order = pd.read_table(pvaluename, index_col=0).iloc[list(range(20000))]
        # print(order)
        X = data.loc[order.index]
        X = X.T
    else:
        X = data.T  # 纵轴为标签
    # use linear regression as the model
    lr = LinearRegression()
    # rank all features, i.e continue the elimination until the last one
    rfe = RFE(lr, n_features_to_select=1)
    rfe.fit(X, label)
    result = pd.DataFrame({geneOrSite: X.columns.values, "rank": rfe.ranking_}).sort_values(by='rank')
    print(select + " result.head()")
    print(result.head())
    col = [geneOrSite, 'rank']
    result.to_csv(tag + '_LR_RFE_rank.txt', sep="\t", index=None, columns=col)
    print(select + " selection finish...")
    print(select + " IFS validation start...")
    result = result.iloc[list(range(cnt))]
    cur_X = data.loc[result[geneOrSite]]
    cur_X.to_csv(tag + "_" + select + "_" + str(cnt) + "_result.csv")
    cur_X = cur_X.T  # 纵轴为标签
    IFS_validation(cur_X, label,times,fold, select,tag)
    print(select + ' IFS validation finish...')
    print("------------------------")

def McOne(data, label, r):
    print("McOne start...")
    classLabel = label
    dataMat = data.values
    n = data.shape[0]
    micFC = [0] * n
    Subset = [-1] * n
    numSubset = 0
    for i in range(n):
        m = MINE()
        m.compute_score(dataMat[i], classLabel)
        micFC[i] = m.mic()
        if micFC[i] >= r:
            Subset[numSubset] = i
            numSubset += 1
    Subset = Subset[:numSubset]
    Subset.sort(key=lambda x: micFC[x], reverse=True)
    e = 0
    while e <= numSubset - 1:
        q = e + 1
        while q <= numSubset - 1:
            m = MINE()
            m.compute_score(dataMat[Subset[e]], dataMat[Subset[q]])
            if m.mic() >= micFC[Subset[q]]:
                for i in range(q, numSubset - 1):
                    Subset[i] = Subset[i + 1]
                numSubset -= 1
            else:
                q += 1
        e += 1
    return data.iloc[Subset[:numSubset]]

def BFS(X, Y):
    print("McTwo BFS start...")
    names_left = list(X.columns)
    flag = 1
    selected_feature = []
    loo = LeaveOneOut()
    max_avc = 0
    while flag != 0:
        flag = 0
        best_feat = 0
        for feat in names_left:
            curr_feature = selected_feature[:]
            curr_feature.append(feat)
            X_temp = X[curr_feature]  # index:sample
            Avc = 0
            for train_index, test_index in loo.split(X_temp):
                TN = 0
                FN = 0
                TP = 0
                FP = 0
                X_train, X_test, Y_train, Y_test = X_temp.iloc[train_index], X_temp.iloc[test_index], Y[train_index], Y[
                    test_index]
                clf = KNeighborsClassifier(n_neighbors=1)
                clf.fit(X_train, Y_train)
                pred = clf.predict(X_test)
                # compute Avc
                # print('pred[0]',pred[0],'Y_test[0]',Y_test[0])
                if ((pred[0] == 1) and (Y_test[0] == 1)):
                    TP = TP + 1
                elif ((pred[0] == 0) and (Y_test[0] == 1)):
                    FN = FN + 1
                elif ((pred[0] == 1) and (Y_test[0] == 0)):
                    FP = FP + 1
                elif ((pred[0] == 0) and (Y_test[0] == 0)):
                    TN = TN + 1
                if ((TP + FN) != 0):
                    Sn = TP / (TP + FN)
                else:
                    Sn = 0
                if ((TN + FP) != 0):
                    Sp = TN / (TN + FP)
                else:
                    Sp = 0
                Avc += ((Sn + Sp) / 2)
                # print('Avc',Avc)
            if Avc > max_avc:
                # print(max_avc)
                # print('maxAvc',Avc)
                max_avc = Avc
                best_feat = feat
                flag = 1
        if (flag == 1):
            selected_feature.append(best_feat)
            names_left.remove(best_feat)
    res = X[selected_feature]
    return res, selected_feature

def McTwo(data, label, r,times,fold,tag):
    print("McTwo start...")
    res = McOne(data, label, r)
    res.to_csv(tag + '_McOne_result.txt', sep="\t")
    print("McOne result")
    print(res.shape)
    X_reduce, feature_name = BFS(res.T, label)
    print("BFS result")
    print(feature_name)
    X_reduce.T.to_csv(tag + "_McTwo_result.txt", sep="\t")
    # X_reduce, feature_name = McTwo(data, label, r)
    # print(X_reduce.shape)
    acc_res = classfiers(X_reduce.values, label,times,fold)
    col = ['SVM', 'GaussianNB', 'KNN', 'DTree', 'GBDT', 'ELM']
    score = pd.DataFrame(acc_res, index=col)
    score.columns = ["Acc"]
    score["feature_number"] = [len(feature_name)] * len(col)
    score.T.to_csv(tag + "_McTwo_acc.csv", columns=col)
    print("McTwo selection finish...")
    print("------------------------")

def select_feature(filename,labelname,pvaluename,r,times,fold,cnt,datasetName,geneOrSite):
    #参数说明：
    #filename:做特征选择的特征数据文件名称，labelname:标签文件名称，pvaluename：TRank后pvalue文件名称，三者都是txt文件
    #r:McTwo（McOne）特征选择的阈值
    #times:交叉验证的次数，fold：交叉验证分的折数
    #cnt:进行IFS的特征数目
    #datasetName:数据集名称(如GSE66695)用于生成结果写文件名称
    #geneOrSite:标明是基因还是位点，若是位点，SVM_RFE,LR_RFE选取pvalue最小的前20000个做特征选择，
    tag = datasetName + "_" + geneOrSite#用于生成结果写文件的前缀
    data, label = load_data(filename, labelname)
    #特征选择列表
    method = [TRank(data, label, pvaluename,times,fold,cnt,tag), WRank(data, label,times,fold,cnt,tag,geneOrSite),
              Chi2(data, label,times,fold,cnt,tag,geneOrSite),
              ROCRank(data, label,times,fold,cnt,tag,geneOrSite),RF(data, label,times,fold,cnt,tag,geneOrSite),
              McTwo(data, label, r,times,fold,tag), SVM_RFE(data, label,times,fold,cnt,tag,geneOrSite,pvaluename),
              LR_RFE(data, label,times,fold,cnt,tag,geneOrSite,pvaluename)]

    for m in method:
        m


