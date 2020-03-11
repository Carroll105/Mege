#encoding=UTF-8
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

def heatmap_gene(fname,name,num,data):
    if name=='McTwo':
        ori_data = pd.read_table(data + '_' + fname + '_' + name + '_' + "100_result.csv", sep=',', index_col=0).iloc[:, :]
        num=len(ori_data.index)
    else:
        ori_data=pd.read_table(data+'_'+fname+'_'+name+'_'+"100_result.csv",sep=',', index_col=0).iloc[0:num, :]
    print(ori_data)
    gene=ori_data.index.tolist()
    ori_data=pd.DataFrame(np.array(ori_data).T)
    ori_data.columns=gene
    print(ori_data)
    sp = pd.read_table("label_test.txt").iloc[:, 1]
    species= pd.Series(sp)
    lut = dict(zip(species.unique(), "rbg"))
    row_colors = species.map(lut)
    print(row_colors)
    g=sns.clustermap(ori_data,figsize=(16,20),row_colors=row_colors)
    g.gs.update(left=0.05)
    #g.gs.update(left=0.05,right=0.45)
    #gs2 = matplotlib.gridspec.GridSpec(1, 1, left=0.6)
    #ax2 = g.fig.add_subplot(gs2[0])
   # sns.boxplot(data=ori_data, orient="h", palette="Set2", ax=ax2)
    plt.title(data+'_'+fname+'_'+name+str(num))
    plt.savefig(data+'_'+fname+'_'+name+'_'+str(num)+".png")
    plt.show()
    return 0

def heatmap(fn_name,data_type,rank_num,data_name):
    #heatmap(['McTwo'],'gene',[10,25,35],'GSE42861')
    # fn_name: ['SVM-RFE']
    # data_type： gene or site
    # rank_num: [10,20,30]
    # data_name: GSE42861
    for i in  range(len(fn_name)):
        name=fn_name[i]
        print(name)
        for mm in range(len(rank_num)):
            heatmap_gene(data_type,name, rank_num[mm], data_name)




