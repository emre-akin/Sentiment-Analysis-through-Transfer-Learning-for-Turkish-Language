import pandas as pd
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
    '''
    pos = pd.read_csv('D:/Emre/source/data_equal/clean/rest_pos_clean_normalized', header=None, names=['text','is_valid'])
    neg = pd.read_csv('D:/Emre/source/data_equal/rest_neg_clean_norm_np', header=None, names=['text','is_valid'])
    negT = neg[:340]
    negV = neg[340:]
    negT['is_valid'] = False
    negV['is_valid'] = True
    neg = pd.concat([negT, negV])

    posT = pos[:340]
    posV = pos[340:]
    posT['is_valid'] = False
    posV['is_valid'] = True
    pos = pd.concat([posT, posV])
    print(pos)
    df_regroup = pd.concat([pos, neg])
    df_regroup.to_csv('D:/Emre/source/data_equal/rest_full.csv', header=None, index=None)
    '''
    '''
    tokenizer = Tokenizer(lang='tr')
    data_lm = TextLMDataBunch.from_csv('D:/Emre/source/data_equal/', csv_name='rest_full.csv',text_cols=0, tokenizer=tokenizer)
    data_lm.save('data_lm_rest_fn.pkl')
    
    data_lm = load_data('D:/Emre/source/data_equal/', 'data_lm_rest_fn.pkl', bs=32)
    learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3 )
    learn.load_pretrained('D:/Emre/source/data/wiki/tr/models/model-full-v2.pth','D:/Emre/source/data/wiki/tr/full_lm_60k/itos.pkl')
    #learn.freeze()
    learn.lr_find()
    learn.recorder.plot(skip_start=15)
    plt.show()

    learn.fit_one_cycle(1, 1e-02)
    learn.save('rest_head_pretrained')

    learn.unfreeze()
    learn.fit_one_cycle(10, 1e-03, moms=(0.8, 0.7))
    learn.save('rest_lm_fine_tuned')
    learn.save_encoder('rest_enc_fine_tuned')
    '''
    
    #data_lm = load_data('D:/Emre/source/data_equal/', 'data_lm_rest_fn.pkl', bs=32)
    #Test
    #data_clas = TextClasDataBunch.from_csv('D:/Emre/source/data_equal/',vocab=data_lm.vocab ,csv_name='rest_full_clas.csv')
    #data_clas.save('rest_data_clas.pkl')
    

    #data_clas = load_data('D:/Emre/source/data_equal/', 'rest_data_clas.pkl', bs=32)
    data_clas = TextClasDataBunch.from_csv('D:/Emre/source/data_equal/',csv_name='rest_full_clas.csv', bs=32)

    learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
    #learn.load_encoder('rest_enc_fine_tuned')
    learn.freeze()
    
    learn.lr_find()
    learn.recorder.plot()
    plt.show()
    print(learn.model)
    learn.fit_one_cycle(1, 1.45e-01, moms=(0.8,0.7))
    #learn.save('rest_first')
    #learn.load('rest_first')

    learn.freeze_to(-2)
    learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))
    #learn.save('rest_second')
    #learn.load('rest_second')

    learn.freeze_to(-3)
    learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))
    #learn.save('rest_third')
    
    #learn.load('rest_third')

    learn.unfreeze()
    learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))
    print(learn.predict("Güzel ürün tavsiye ederim."))
    print(learn.predict("Kötü"))
    print(learn.predict("Rezalet ötesi"))
    
    

if __name__ == "__main__":
    run()