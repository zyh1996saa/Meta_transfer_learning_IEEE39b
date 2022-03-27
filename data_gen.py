import pypower
from pypower import case39 
from pypower import runpf
import numpy as np
import copy
import random
from pypower.makeSbus import makeSbus
from pypower.ext2int import ext2int
from pypower.loadcase import loadcase
from pypower.makeYbus import makeYbus
import time

case = case39.case39()
TOPO_num = 1
PF_sample_num = 1000
BRANCH_num = case['branch'].shape[0]
BUS_num = case['bus'].shape[0]
GEN_num = case['gen'].shape[0]
MAX_disconn = 2
randon_power_coef = [0.95,1.05]

def makeY(mod_case):
    ppc = ext2int(mod_case)
    baseMVA, bus, gen, branch = \
        ppc["baseMVA"], ppc["bus"], ppc["gen"], ppc["branch"]
    Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
    Y = np.zeros((BUS_num,BUS_num),dtype='complex64')
    for i in range(BUS_num):
        for j in range(BUS_num):
            Y[i,j] = Ybus[i,j]
    return Y

def makeS(mod_case):
    ppc = ext2int(mod_case)
    baseMVA, bus, gen, branch = \
        ppc["baseMVA"], ppc["bus"], ppc["gen"], ppc["branch"]
    Sbus = makeSbus(baseMVA, bus, gen)
    S = np.zeros((BUS_num,1),dtype='complex64')
    for i in range(BUS_num):
        S[i,0] = Sbus[i]
    return S

def makeU(mod_res):
    Vm = np.zeros((BUS_num,1))
    Va = np.zeros((BUS_num,1))
    for i in range(BUS_num):
        Vm[i,0] = mod_res['bus'][i,7]
        Va[i,0] = mod_res['bus'][i,8]
    return Vm,Va
        
        
def get_del_branch(disconn_num):
    del_branch_nums = []
    for i in range(disconn_num):
        del_num = random.randint(0,BRANCH_num-1)
        if del_num not in del_branch_nums:
            del_branch_nums.append(del_num)
        else:
            continue
    return del_branch_nums

def random_PF(temp_case):
    mod_case = copy.deepcopy(temp_case)
    max_try = 100
    cur_try = 0
    while 1:
        if cur_try>max_try:break
        for bus_label in range(BUS_num):
            mod_case['bus'][bus_label,2] *=  random.uniform(randon_power_coef[0],randon_power_coef[1])
            mod_case['bus'][bus_label,3] *=  random.uniform(randon_power_coef[0],randon_power_coef[1])
        for gen_label in range(GEN_num):
            mod_case['gen'][gen_label,1] *=  random.uniform(randon_power_coef[0],randon_power_coef[1])
            mod_case['gen'][gen_label,2] *=  random.uniform(randon_power_coef[0],randon_power_coef[1])
        res = pypower.runpf.runpf(mod_case)
        if res[1] == 1:
            res = res[0]
            break
        else:
            #break
            continue
    return mod_case, res



if __name__ == '__main__':
    #for topo_num in range(TOPO_num):
    topo_num = 0    
    #total_dict = {}
    while 1:   
        print('\r%s/%s'%(topo_num,TOPO_num),end='\r')
        if topo_num >= TOPO_num: break
        topo_dict = {}
        temp_case = copy.deepcopy(case)
        disconn_num = MAX_disconn
        del_branch_nums = get_del_branch(disconn_num)
        temp_case['branch'] = np.delete(temp_case['branch'],del_branch_nums, axis = 0)
        if pypower.runpf.runpf(temp_case)[1] == 0:continue
        #topo_dict['Y'] = makeY(temp_case)
        topo_dict['del_branches'] = del_branch_nums
        topo_dict['PF_samples'] = {}
        for inner_loop_num in  range(PF_sample_num):  
            print('\r%s/%s'%(inner_loop_num,PF_sample_num),end='\r')
            mod_case,mod_res = random_PF(temp_case)
            topo_dict['PF_samples'][inner_loop_num] = {}
            topo_dict['PF_samples'][inner_loop_num]['S'] = makeS(mod_case)
            topo_dict['PF_samples'][inner_loop_num]['Vm'],topo_dict['PF_samples'][inner_loop_num]['Va'] = makeU(mod_res)
        #total_dict['topo'+str(topo_num)] = topo_dict
        topo_num += 1
        np.save(r"./data/topo%s.npy"%topo_num,topo_dict)
        
    #np.save("total_data.npy",total_dict)