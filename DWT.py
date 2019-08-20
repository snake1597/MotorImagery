import csv, copy, os
import numpy as np
import pywt
from scipy import signal
from sklearn import preprocessing

def feature_extraction(x_data):
    DWTfamilyName = 'db4'
    coeffs = pywt.wavedec(x_data, DWTfamilyName, level=5, axis=0)
    x_train = preprocessing.normalize(coeffs[0], axis=1)
    return x_train

count = 0
subNames = 'testData'.split(' ')
directory = './bci2003/testData_DWT'
storetotxt = './bci2003/testData_DWT_fileName.txt'
write_txt = open(storetotxt, 'w')
if not os.path.exists(directory):
    os.makedirs(directory)

for subname in subNames:
    txtpath = "./bci2003/{0}_fileName.txt".format(subname)
    with open(txtpath) as f:
        content = f.readlines()
        File_list = list(content)
        for file_name in content:
            file_name = file_name.strip()
            f_namelist = file_name.split('_')
            csvpath = "./bci2003/{0}/{1}".format(subname,file_name)
            csvfile = open(csvpath, 'r')
            csv_list = list(csv.reader(csvfile))
            data = list()
            # select 4.5 seconds data           
            for i in range(0,576):
                data.append(csv_list[i])
            data = np.array(data,dtype=float)
            data = np.power(data, 2)
            # do DWT
            result = feature_extraction(data)    
            count += 1
            DWT_file = directory+'/'+file_name
            myfile = open(DWT_file, 'w', newline='')
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerows(result)
            myfile.close()
            con = file_name
            write_txt.write(con+'\n')
write_txt.close()
print('end')