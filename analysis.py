import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import os

def step_key(file_name):
    return int(file_name.split('_')[1])

N=200
F=0.001

loc="/data/kabir/output/SOC_model/data_f"+str(F)+"N"+str(N)+"/" #180519_
files=os.listdir(loc)
files.remove("avalanche_sizes.pkl")
for file in files:
    if "time" in file:
        time_file=file
#print(time_file)
files.remove(time_file)
files=sorted(files,key=step_key)
print(files)

steps=[]
data=[]
for file in files:
    path=loc+file
    #print(file)
    with open(path,'rb') as pickle_file:
        u = pkl._Unpickler(pickle_file)
        u.encoding = 'latin1'
        data.append(u.load()['dist'])
    steps.append(step_key(file))
"""
with open(loc+'avalanche_sizes.pkl','rb') as pickle_file:
        u = pkl._Unpickler(pickle_file)
        u.encoding = 'latin1'
        avalanche_sizes=u.load()
#"""
step=-1 ###last step is -1, first step is 0
fig = plt.figure(figsize=(15, 7))
plt.plot(data[step][0],data[step][1],'o')

plt.yscale('log')
plt.xscale('log')
