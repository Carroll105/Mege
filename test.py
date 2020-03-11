#test.py
from Mege import GSEData_preprocess
from Mege import GSEData_fit

if __name__ == '__main__':
    GSEData_preprocess('GSE42861','\"Patient genomic DNA from sample 1\"','\"Normal genomic DNA from sample 52\"','!Sample_title')
    #GSEData_preprocess(数据集编号,原文件正样本标签,原文件负样本标签,样本标签行起始标志)
    GSEData_fit('GSE42861','LinearRegression','origin_data')
    #GSEData_fit(数据集编号,拟合模型,数据类型)
