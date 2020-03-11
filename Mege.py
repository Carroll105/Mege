#data_fit.py
import re
import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

dataset_list=['GSE103186','GSE53045','GSE74845','GSE80417'] #for test
    
def origin_data(data):
    return data
def square_data(data):
    return data**2
def log_data(data):
    return np.log(data+1e-5)
def radical_data(data):
    return data**(1/2)
def cube_data(data):
    return data**3

def divide(code,pname,nname,rowhead):
    org_file=open('.//'+code+'//'+code+"_series_matrix.txt","r")
    sample_info=open('.//'+code+'//'+code+"_info.txt","w+")
    matrix=open('.//'+code+'//'+code+".txt","w+")
    character=open('.//'+code+'//'+code+"_charact.txt","w+")
    classlabel=[]

    #title
    for line in org_file:
        if re.match('!Series',line):
            sample_info.write(line)

        if re.match(rowhead,line):
            label=line.split("\t")
            for item in label:
                if item==(pname):
                    classlabel.append(0)
                elif item==(nname):
                    classlabel.append(1)
            pd.DataFrame(classlabel).to_csv('.//'+code+'//'+code+"_label.txt",sep='\t')#,"w+"
            print("Label read in finish!")
        if re.match('!Sample_characteristics_ch1',line):
            character.write(line)
        if re.match("!series_matrix_table_begin",line):
            print("Matrix read in begin!")
            break

    #content
    count=0
    for line in org_file:
        if re.match("!series_matrix_table_end",line):
            break
        matrix.write(line)
        count+=1
        print("line "+count.__str__()+" read in finish!")
    print("Matrix read in finish!")
    org_file.close()
    sample_info.close()
    matrix.close()
    character.close()
    print("Finish!")
    
def pnsplit(x_data,y_data,tratio):#纵向为样本，横向为标签
    x_po_data=[]
    x_ne_data=[]
    y_po_data=[]
    y_ne_data=[]
    for i in range(len(y_data)):
        if y_data[i]==0:
            x_ne_data.append(x_data[:,i])
            y_ne_data.append(0)
        if y_data[i]==1:
            x_po_data.append(x_data[:,i])
            y_po_data.append(1)
    x_po_data=np.array(x_po_data)
    x_ne_data=np.array(x_ne_data)
    y_po_data=np.array(y_po_data)
    y_ne_data=np.array(y_ne_data)
    x_po_train,x_po_test,y_po_train,y_po_test=train_test_split(x_po_data,y_po_data[:,np.newaxis],test_size=tratio,random_state=1)
    x_ne_train,x_ne_test,y_ne_train,y_ne_test=train_test_split(x_ne_data,y_ne_data[:,np.newaxis],test_size=tratio,random_state=1)
    x_train=np.vstack((x_po_train,x_ne_train))
    y_train=np.vstack((y_po_train,y_ne_train))
    x_test=np.vstack((x_po_test,x_ne_test))
    y_test=np.vstack((y_po_test,y_ne_test))
    return x_train.T,x_test.T,y_train,y_test

def delete_nan(code,data):
    print("Now deleting nan data...")
    num=len(data.index)
    nanspot=[]
    nansample=[]
    for spot in range(len(data.index)):
        print("Now processing "+data.index[spot]+"\t"+str(int(spot/num)*100)+"% ...")
        for sample in range(len(data.columns)):
            if np.isnan(data.iloc[spot][sample]):
                nanspot.append(data.index[spot])
                nansample.append(data.columns[sample])
        print("finish!")
    realspot=[]
    for spot in data.index:
        if spot not in nanspot:
            realspot.append(spot)
    pd.DataFrame(nansample,index=nanspot,columns=['nansample']).to_csv('.//'+code+'//result//nansample.txt',sep='\t')
    data=data.loc[realspot]
    data.to_csv('.//'+code+'//'+code+'_.txt',sep='\t')
    print("Deleting Finish!")
    return data

def sample_split(code,data,label):
    sample_list=data.columns
    spot_list=data.index
    print("Index load in finish!")
    sample_train,sample_test,label_train,label_test=pnsplit(np.array(sample_list)[np.newaxis,:],np.array(label),0.33)
    pd.DataFrame(sample_train).to_csv('.//'+code+'//result//sample_train.txt',sep='\t')
    pd.DataFrame(sample_test).to_csv('.//'+code+'//result//sample_test.txt',sep='\t')
    pd.DataFrame(label_train).to_csv('.//'+code+'//result//label_train.txt',sep='\t')
    pd.DataFrame(label_test).to_csv('.//'+code+'//result//label_test.txt',sep='\t')
    print('Sample split finsh!')
    sampletrain=sample_train
    sampletest=sample_test
    sample_train=[]
    sample_test=[]
    for sample in sampletrain[0]:
        sample_train.append(sample)
    for sample in sampletest[0]:
        sample_test.append(sample)
    data_train=data.loc[:,sample_train]
    data_test=data.loc[:,sample_test]
    data_train.to_csv('.//'+code+'//result//data_train.txt',sep='\t')
    data_test.to_csv('.//'+code+'//result//data_test.txt',sep='\t')
    return spot_list,sample_train,sample_test,label_train,label_test,data_train,data_test

def ori_pvalue(code,data_test,label_test):
    pdata=[]
    ndata=[]
    pvalue=[]
    print('Now start divide n/p data(spot)...')
    count=0
    num=len(data_test.columns)
    for sample in range(len(label_test)): 
        count+=1
        print('Now processing sample: '+str(int(count*100/num))+"%")
        if label_test[sample]==0:
            ndata.append(data_test.iloc[:,sample])
        if label_test[sample]==1:
            pdata.append(data_test.iloc[:,sample])
        print('finish!')
    print('Process Finish!')
    pdata=np.array(pdata).T
    ndata=np.array(ndata).T
    print('Now start calculating pvalue(spot)...')
    count=0
    num=len(data_test.index)
    for spot in range(len(data_test.index)):
        count+=1
        print('Now calculating spot: '+str(int(count*100/num))+"%")
        ttest=stats.ttest_ind(ndata[spot,:],pdata[spot,:])
        pvalue.append(ttest.pvalue)
        print('finish!')
    print('Process Finish!')
    result=pd.DataFrame(pvalue,index=data_test.index,columns=['pvalue'])
    result=result.sort_values(by=['pvalue'],ascending=True)
    result.to_csv('.//'+code+'//result//data_test_pvalue.txt',sep='\t')

def GSEData_fit(code,model_type,data_type):
    data_dict={'origin_data':origin_data,'square_data':square_data,'log_data':log_data,'radical_data':radical_data,'cube_data':cube_data}
    model_dict={'LinearRegression':LinearRegression,'LogisticRegression':LogisticRegression,'L1':Lasso,'L2':Ridge}
    data=pd.read_csv('.//'+code+'//'+code+'_.txt',sep='\t',index_col='ID_REF')
    print("Data load in finish!")
    with open('Gene_dict.json','r') as f:
        gene_dict=json.load(f)
        f.close()
    print("Index load in finish!")
    sample_train=pd.read_csv('.//'+code+'//result//sample_train.txt',sep='\t',index_col='Unnamed: 0')
    sample_test=pd.read_csv('.//'+code+'//result//sample_test.txt',sep='\t',index_col='Unnamed: 0')
    label_train=pd.read_csv('.//'+code+'//result//label_train.txt',sep='\t').iloc[:,1]
    label_test=pd.read_csv('.//'+code+'//result//label_test.txt',sep='\t').iloc[:,1]
    print("Label load in finish!")
    
    count=0
    num=len(gene_dict)
    gene_list=[]
    print('Now start learning gene...')
    #starttime=time.time()
    #for data_type in data_dict:
    for gene in gene_dict:
        count+=1
        data_train=data_dict[data_type](data.loc[:,sample_train.values[0]])
        data_test=data_dict[data_type](data.loc[:,sample_test.values[0]])
        gene_data_train=[]
        gene_data_test=[]
        for spot in data_train.index:
            if spot in gene_dict[gene]:
                gene_data_train.append(data_train.loc[spot])
                gene_data_test.append(data_test.loc[spot])
        if len(gene_data_train)==0 or len(gene_data_test)==0: 
            print('Contained Nan data, has been removed!')
            continue
        gene_data_train=np.array(gene_data_train)
        gene_data_test=np.array(gene_data_test)
        gene_list.append(gene)
        print('Now processing '+gene+"\tusing "+model_type+"\ton "+data_type+"\t"+str(int(count*100/num))+'% ...')
        model=model_dict[model_type]()
        model.fit(gene_data_train.T,label_train)
        pred2=model.predict(gene_data_test.T)
        if count == 1:
            data_test_pred=pred2.T
        else:
            data_test_pred=np.vstack([data_test_pred,pred2.T])
        print('finish!')
    data_test_pred=pd.DataFrame(np.array(data_test_pred),index=gene_list)
    data_test_pred.to_csv('.//'+code+'//result//data_test_pred('+data_type+'_'+model_type+').txt',sep='\t')
    print("Learning finish!")
    
    pdata=[]
    ndata=[]
    pvalue=[]
    print('Now start divide n/p data(gene)...')
    count=0
    num=len(data_test_pred.columns)
    for sample in range(len(label_test)): 
        count+=1
        print('Now processing sample: '+str(int(count*100/num))+"%")
        if label_test[sample]==0:
            ndata.append(data_test_pred.iloc[:,sample])
        if label_test[sample]==1:
            pdata.append(data_test_pred.iloc[:,sample])
        print('finish!')
    print('Process Finish!')
    pdata=np.array(pdata).T
    ndata=np.array(ndata).T
    print('Now start calculating pvalue(gene)...')
    count=0
    num=len(data_test_pred.index)
    for gene in range(len(data_test_pred.index)):
        count+=1
        print('Now calculating gene: '+str(int(count*100/num))+"%")
        ttest=stats.ttest_ind(ndata[gene,:],pdata[gene,:])
        pvalue.append(ttest.pvalue)
        print('finish!')
    print('Process Finish!')
    result=pd.DataFrame(pvalue,index=data_test_pred.index,columns=['pvalue'])
    result=result.sort_values(by=['pvalue'],ascending=True)
    
    result.to_csv('.//'+code+'//result//data_test_pred_pvalue('+data_type+'_'+model_type+').txt',sep='\t')
    print("================FINISH!================")

def GSEData_preprocess(code,pname,nname,rowhead):
    try:
        os.mkdir('.//'+code+'//')
    except:
        print('Folder already exist!')
    try:
        os.mkdir('.//'+code+'//result//')
    except:
        print('Folder already exist!')
    divide(code,pname,nname,rowhead)
    data=pd.read_csv('.//'+code+'//'+code+'.txt',sep='\t',index_col='ID_REF')
    label=pd.read_csv('.//'+code+'//'+code+"_label.txt",sep='\t').iloc[:,1]
    data=delete_nan(code,data)
    spot_list,sample_train,sample_test,label_train,label_test,data_train,data_test=sample_split(code,data,label)
    ori_pvalue(code,data_test,label_test)
    
    
    
    
