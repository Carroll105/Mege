#encoding=UTF-8
import scipy
import scipy.cluster.hierarchy as sch
from scipy.cluster.vq import vq, kmeans, whiten
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt
import scipy.spatial as ch
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn import cluster

#定义purity


def purity(label_set):
    num=0
    for i in range(len(label_set)):
        if label_set[i]==0:
            num=num+1
    return max(num/len(label_set),1-num/len(label_set))

#定义purity计算
def purity_cal(data,n,st,sp,list_pu,num):
    if st=='McTwo':
        ori_data = pd.read_table(data + '_' + sp + '_' + st + "_100_result" + ".csv", sep=',', index_col=0).iloc[:, :]
    else:
        ori_data=pd.read_table(data+'_'+sp+'_'+st+"_100_result"+".csv",sep=',',index_col=0).iloc[0:n,:]
    #ori_data = pd.read_table(data + sp + st + "result" + ".txt", index_col=0).iloc[:, :]
    #ori_data=ori_data.loc[gene]
    ori_data=pd.DataFrame(np.array(ori_data).T)
    #sp = pd.read_table("label2.txt").iloc[:, 1]
    sp = pd.read_table("label_test.txt").iloc[:, 1]
    label_num=len(sp.index.tolist())
    ori_data=np.array(ori_data)
    #1. 层次聚类
    #生成点与点之间的距离矩阵,这里用的欧氏距离:
    disMat = ch.distance.pdist(ori_data, 'euclidean')
    disMat = ch.distance.squareform(disMat)
    #print(disMat)
    #进行层次聚类:
    #print("%d clusters" % num_clusters)
    X, labels_true = disMat,sp.tolist()
    labels_num=len(labels_true)
    num_firest=0
    for i in range(len(labels_true)):
        if labels_true==0:
            num_firest=num_firest+1
    the_first_cluster=max(num_firest/labels_num,1-num_firest/labels_num)
    print('only one cluster',the_first_cluster)
    print(labels_num)
    for i in range(2,31):
        clst = cluster.AgglomerativeClustering(n_clusters=i)
        predicted_labels = clst.fit_predict(X)
        #score=adjusted_rand_score(labels_true, predicted_labels)
        score_pu=[]
        len_li=[]
        for h in range(0, i):
            sc,nu=purity(labels_true,predicted_labels, h)
            score_pu.append(sc)
            len_li.append(nu)
        all_purity=0
        print("score")
        print(score_pu,len_li)
        for nn in range(len(score_pu)):
            if len(score_pu)==1:
                all_purity=the_first_cluster
            else:
                all_purity=all_purity+(score_pu[nn]*len_li[nn])/labels_num
        list_pu.append(all_purity)
        num.append(i)
    return list_pu,num

def purity(la,lb,x):
    list=[]
    num=0
    for i in range(len(lb)):
        if lb[i] == x:
            list.append(la[i])
    for i in range(len(list)):
        if list[i]==0:
            num=num+1
    return max(num/len(list),1-num/len(list)),len(list)


def hierarhy_analy(function_name,rank_numlist,data_name,data_type):
    #list_data = 'SVM_RFE_50_,LR_RFE_50_,RF_50_,ttest_pvalue_50_,wilcoxon_test_pvalue_50_,GaussianNB_AUC_50_'
    #list_data = 'RF_50_,TRank_50_,WRank_50_,ROCRank_50_,chi2_50_'
    #list_data = '_SVM_RFE_,_LR_RFE_,_RF_,_TRank_,_WRank_,_ROCRank_,_Chi2_'

    #part = ['origin_L2_gene']

    # function_name:SVM_RFE
    # rank_numlist:[10,20,30]
    # data_name:"GSE42861"
    # data_type:'gene'
    for ia in  range(len(function_name)):
        res = pd.DataFrame()
        for ib in range(len(rank_numlist)):
            list_pu = []
            num = []
            list_pu,num=purity_cal(data_name,rank_numlist[ib],function_name[ia],data_type,list_pu,num)
            print(list_pu,num)
            pd_gene=pd.DataFrame(list_pu,index=range(2,31))
            # list_pu = []
            # num = []
            # list_pu,num=purity_cal(na[ib],list_data[ia],'spot')
            # pd_spot=pd.DataFrame(list_pu,columns=['spot'],index=range(2,31))
            res=pd.concat([res,pd_gene],axis=1)
        print(res)
        res.columns = rank_numlist
        print(res)
        pd.DataFrame(res).to_csv(data_name+'_'+data_type+'_'+function_name[ia]+"_purity"+".csv")

if __name__ == '__main__':
    # function_name = "_McTwo_"
    # #list_data = '_SVM_RFE_'
    #
    # rank_numlist=[10,20,30,40,50]
    # #na=[10]
    # data_name="GSE42861"
    # part='gene'
    hierarhy_analy(['McTwo','TRank'],[10,20,30],"GSE42861",'gene')

