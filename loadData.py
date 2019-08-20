import csv, random
import numpy as np

class load:
    def __init__(self, subNames, path='./data', notEqual='NaN', isRand=False):
        self.subNames = subNames
        self.path     = path
        self.isRand   = isRand
        self.notEqual = notEqual

    def loadTrainDataFromTxt(self):
        y_trainList = list()
        x_trainList = list()
        subNames = self.subNames
        for subname in subNames:
            txtPath = '{0}/{1}_fileName.txt'.format(self.path,subname)
            with open(txtPath) as f:
                content = f.readlines()
                File_list = list(content)
                for f_name in File_list:
                    f_name = f_name.strip()
                    f_namelist = f_name.split('_')
                    if f_namelist[1] != self.notEqual:            
                        csvPath = '{0}/{1}/{2}'.format(self.path, subname, f_name)
                        csvfile = open( csvPath, 'r')
                        csv_list = list(csv.reader(csvfile))
                        x_trainList.append(csv_list)
                        y_trainList.append(f_namelist[1])

        if self.isRand:
            index_shuf  = list(zip(x_trainList, y_trainList))
            random.shuffle(index_shuf)
            x_trainList, y_trainList= zip(*index_shuf)
    
        x_trainList = np.array(x_trainList,dtype=float)
        y_trainList = np.array(y_trainList,dtype=int)
        return x_trainList, y_trainList