#!/usr/bin/env python3
import subprocess
import sys
import os
import multiprocessing as mp
from multiprocessing import Pool

cwd = os.getcwd()
file_name = os.path.join(cwd, "experiment_gsvm.param") 

#1. Parameter files
with open(file_name) as f:
    params = f.readlines()

#2. Parameter list
param_list = []
for param in params:
    parameters = param.split()
    parameters[-1] = os.path.join(cwd, "datasets/imbadatanorm/", parameters[-1])
    param_list.append(parameters)
# print(param_list)

# #3. Subprocess
param_list_2 = []
for param in param_list:
    executable = os.path.join(cwd, "build/app/gsvm")
    param = [executable] + param
    param_list_2.append(param)
# print(param_list_2)

def execute(param):
    fname = param[-1].split("/")
    dataset = fname[-1]
    # proc = subprocess.call(param, stdout=subprocess.PIPE)
    result_str = subprocess.check_output(param)
    return (dataset,result_str)

# print(mp.cpu_count())
results = []
if __name__ == '__main__':
    with Pool(mp.cpu_count()) as p:
        results = p.map(execute, param_list_2)
    with open('listresult.txt', 'w') as filehandle:
        for listitem in results:
            filehandle.write(str(listitem))