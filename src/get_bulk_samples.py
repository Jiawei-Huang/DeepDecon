import random
import os
import pandas as pd
import numpy as np
import argparse

def getSample(x, num=None):
    if x is None:
        return 0, None
    n = x.shape[0]
    if num is not None and num>=0:
        select_num = num
    else:
        select_num = random.randint(0, n)
    if select_num == 0:
        return 0, None
    repeat = False if select_num<=n else True # Allow or disallow sampling of the same row more than once.
    select = x.sample(select_num, replace=repeat)
    
    return select_num, select.sum(axis=0)

def getFixedBulksample(x, fixed, N=100, total=200):
    malig = np.array([int(N*fixed['malignant'])]*total)
    count = np.array([malig, N-malig]).T
    gene = [x[ x['cellType']==ct ].iloc[:, 1:] for ct in ['malignant', 'normal'] ]
    res = []
    for i in range(total):
        sample = []
        for j in range(len(count[i])):
            sub = gene[j]
            num = count[i][j]
            repeat = False if num<=len(sub) else True
            select = sub.sample(num, replace=repeat)
            select = select.sum(axis=0)
            sample.append(select)
        res.append(pd.DataFrame(sample).sum(axis=0))
    return pd.DataFrame(res)
   
def getBulksample(x, N=100, total=None, normal=False, binomial=False, fixed = None, ratio_range=None):
    """
    x : sc-RNA gene expression #cell * #gene
    N : # cells in one bulk sample
    total : total number of bulk samples
    normal : if True, then N if obtained from Normal(N, 0.05N)
    """
    gene = x
    cellexps = [] # gene expression of a given celltype, e.g. alpha
    cellnames = [] # celltype names, corrsponding to cellexps
    for name, group in gene.groupby('cellType'):
        cellnames.append(name)
        if group.shape[0]>0:
            cellexps.append(group.iloc[:, 1:])
        else:
            cellexps.append(None)
    # n = # bulk sample
    n = total
    exps = [] # bulk gene expression, n*(#gene)
    cts = [] # true number of cells n*(#celltype)
    ratios = [] # true fractions n*(#celltype)
    
    if normal:
        amount = np.rint(np.random.normal(N, 0.05*N, n)).astype(int)
    elif fixed is not None:
        count = []
        for name in cellnames:
            num = np.array([int(N*fixed[name])]*total)
            count.append(num)
        amount = np.array(count).T
    else:
        amount = [N]*n
    
    for idx in range(n):
        ct = []
        exp = []
        remain = amount[idx]
        if binomial:
            if ratio_range is not None and ratio_range[0]==ratio_range[1]:
                ratio = ratio_range[0]/100
                cell_nums = np.random.multinomial(N, [ratio, 1-ratio])
                
            else:
                ratio = np.random.random_sample() 
                cell_nums = np.random.multinomial(remain, [ratio, 1-ratio])
        for i in range(len(cellnames)):
            if binomial:
                num = cell_nums[i]
            elif fixed is not None:
                num = remain[i]
            else:
                if i == len(cellnames)-1:
                    num = remain
                else:
                    if ratio_range is not None:
                        low, high = ratio_range
                        # num = int(remain*random.randint(int(low), int(high))/100)
                        num = int(random.randint(remain*int(low), remain*int(high))/100)
                    else:
                        num = random.randint(0, remain)
                    remain -= num
            cnt, select = getSample(cellexps[i], num)
            
            ct.append(cnt)
            if select is not None:
                exp.append(select)
        exp = pd.DataFrame(exp)
        cts.append(ct)
        exps.append(exp.sum(axis=0))
    
    cts = pd.DataFrame(cts, columns=cellnames)
    exps = pd.DataFrame(exps)
#     print('final', cts, exps)
    return cts, exps

def simulateBulksample(expression, N=100, total=None, normal=False, binomial=False, fixed=None, ratio_range=None):
    frac, x = getBulksample(expression, N, total=total, normal=normal, binomial=binomial, fixed=fixed, ratio_range=ratio_range)
    final = pd.concat([frac, x], axis=1)
    return final

parser = argparse.ArgumentParser()
parser.add_argument("--cells", type=int, help="Number of cells to use for each bulk sample.", default=100)
parser.add_argument("--samples", "-n", type=int, help="Total number of samples to create for each dataset.", default=200)
parser.add_argument("--subject", type=str, help="Subject name", default='AML328-D29')
parser.add_argument("--start", type=int, help="Fraction start range of generated samples e.g. 0 for [0, 100]", default=0)
parser.add_argument("--end", type=int, help="Fraction end range of generated samples e.g. 0 for [0, 100]", default=100)
parser.add_argument("--normal", type=int, help="Whether generating bulk sample from normal distribution, 0=False, 1=True", default=0)
parser.add_argument("--binomial", type=int, help="Whether generating bulk fractions from binomial distribution, 0=False, 1=True", default=0)
parser.add_argument("--data", type=str, help="Directory containg the datsets", default='/project/fsun_106/jiaweih/AML/10x_dataset/gdc_data/aml_subject_data/')
parser.add_argument("--out", type=str, help="Output directory", default='/project/fsun_106/jiaweih/AML/10x_dataset/gdc_data/aml_bulk_simulation_binomial/')
args = parser.parse_args()

sample_size = args.cells
num_samples = args.samples
path = args.data
name = args.subject
normal = True if args.normal == 1 else False
binomial = True if args.binomial == 1 else False
ratio_range = [args.start, args.end]


out_dir = args.out
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
def main():
    tmp = pd.read_csv(path+name+'_norm_sc.txt', index_col=0)
    final = simulateBulksample(tmp, N=sample_size, total=num_samples, normal=normal, binomial=binomial, ratio_range=ratio_range)
    if binomial:
        filepath = out_dir+'fixed_' + str(args.start)+'_'+str(100-args.start)+'_'+str(args.cells)+'_'+str(num_samples)+'_binomial.txt'
    else:
        filepath = out_dir+'_bulk_nor_'+ str(sample_size) + '_'+str(num_samples)+'.txt'
    final.to_csv(filepath)
    print(str(final.shape[0]) + ' bulk samples write to ' + filepath)    
    
if __name__ == "__main__":
    main()
        
    
    