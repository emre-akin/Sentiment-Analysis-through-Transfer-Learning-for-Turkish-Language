import pandas as pd
import numpy as np
from fastai.text import *
import matplotlib.pyplot as plt

def freeze_support():
    ''' Check whether this is a fake forked process in a frozen executable. If so then run code specified by commandline and exit. ''' 
    if sys.platform == 'win32' and getattr(sys, 'frozen', False):
        from multiprocessing.forking import freeze_support
        freeze_support()

def run():
    freeze_support()
    print('loop')
    aa = open('D:/Emre/source/data_equal/rest.txt',"r", encoding="utf-8-sig")
    text = aa.read()
    words = text.split()
    print(len(words))
    average = sum(len(word) for word in words) / len(words)
    print(average)

    infile = open('D:/Emre/source/data_equal/rest.txt',"r", encoding="utf-8-sig")
    lines=0
    words=0
    characters=0
    for line in infile:
        line = line.strip(os.linesep)
        wordslist=line.split()
        lines=lines+1
        words=words+len(wordslist)
        characters=characters+ len(line)
    print(lines)
    print(words)
    print(characters)
    '''
    #For Fine-tuning
    pos = pd.read_csv('D:/Emre/source/reviewsposclean.csv', header=None, names=['text', 'is_valid'])
    neg = pd.read_csv('D:/Emre/source/reviewsnegclean2.csv', header=None, names=['text', 'is_valid'])
    print(neg.__len__())
    neg.dropna()
    print(neg.__len__())
    print(neg)
    neg['is_valid'] = False
    pos['is_valid'] = False
    neg = neg.sample(frac=0.25)
    print(neg.__len__())
    
    negT = neg.sample(frac=0.90)
    negV = neg.drop(negT.index)
    negV['is_valid'] = True

    pos = pos[:15000]
    pos = pos.sample(frac=0.25)
    print(pos.__len__())
    posT = pos.sample(frac=0.90)
    posV = pos.drop(posT.index)
    posV['is_valid'] = True
    
    pos = pd.concat([posT, posV])
    neg = pd.concat([negT, negV])
    df_regroup = pd.concat([pos, neg])
    df_regroup = df_regroup[['text', 'is_valid']]
    df_regroup.to_csv('D:/Emre/source/data_equal/shop25_full.csv', index=None)
    
    #For Classifier
    pos = pd.read_csv('D:/Emre/source/reviewsposclean.csv', header=None, names=['text', 'label', 'is_valid'])
    neg = pd.read_csv('D:/Emre/source/reviewsnegclean2.csv', header=None, names=['text', 'label', 'is_valid'])
    pos['label'] = 'Positive'
    neg['label'] = 'Negative'
    neg['is_valid'] = False
    pos['is_valid'] = False
    neg = neg.sample(frac=0.25)
    negT = neg.sample(frac=0.90)
    negV = neg.drop(negT.index)
    negV['is_valid'] = True

    pos = pos[:15000]
    pos = pos.sample(frac=0.25)
    posT = pos.sample(frac=0.90)
    posV = pos.drop(posT.index)
    posV['is_valid'] = True
    
    pos = pd.concat([posT, posV])
    neg = pd.concat([negT, negV])
    df_regroup = pd.concat([pos, neg])
    df_regroup = df_regroup[['label', 'text', 'is_valid']]
    df_regroup.to_csv('D:/Emre/source/data_equal/shop25_full_clas.csv', index=None)
    '''
    '''
    pos = pd.read_csv('D:/Emre/source/data_equal/film_pos_clean_norm_np', header=None, names=['text', 'is_valid'])
    neg = pd.read_csv('D:/Emre/source/data_equal/film_neg_clean_norm_np', header=None, names=['text', 'is_valid'])
    neg['is_valid'] = False
    pos['is_valid'] = False
    
    negT = neg.sample(frac=0.90)
    negV = neg.drop(negT.index)
    negV['is_valid'] = True

    posT = pos.sample(frac=0.90)
    posV = pos.drop(negT.index)
    posV['is_valid'] = True
    
    pos = pd.concat([posT, posV])
    neg = pd.concat([negT, negV])
    df_regroup = pd.concat([pos, neg])
    df_regroup = df_regroup[['text', 'is_valid']]
    df_regroup.to_csv('D:/Emre/source/data_equal/film_full.csv', index=None)

    #For Classifier
    pos = pd.read_csv('D:/Emre/source/data_equal/film_pos_clean_norm_np', header=None, names=['text', 'label', 'is_valid'])
    neg = pd.read_csv('D:/Emre/source/data_equal/film_neg_clean_norm_np', header=None, names=['text', 'label', 'is_valid'])
    pos['label'] = 'Positive'
    neg['label'] = 'Negative'
    neg['is_valid'] = False
    pos['is_valid'] = False
    
    negT = neg.sample(frac=0.90)
    negV = neg.drop(negT.index)
    negV['is_valid'] = True

    posT = pos.sample(frac=0.90)
    posV = pos.drop(negT.index)
    posV['is_valid'] = True
    
    pos = pd.concat([posT, posV])
    neg = pd.concat([negT, negV])
    df_regroup = pd.concat([pos, neg])
    df_regroup = df_regroup[['label', 'text', 'is_valid']]
    df_regroup.to_csv('D:/Emre/source/data_equal/film_full_clas.csv', index=None)
    '''

if __name__ == "__main__":
    run()