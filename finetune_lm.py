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

    PathModel = Path('Path to your pre-trained model')
    PathCsv = Path('Path to your csv files')
    
    tokenizer = Tokenizer(lang='xx')
    data_lm = TextLMDataBunch.from_csv(PathCsv, csv_name='rest_full.csv',text_cols=0, tokenizer=tokenizer)
    data_lm.save('data_lm_rest_fn.pkl')
    
    data_lm = load_data(PathCsv, 'data_lm_rest_fn.pkl', bs=32)
    learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3 )
    learn.load_pretrained(PathModel'models/model-full-v2.pth',PathModel'full_lm/itos.pkl')
    learn.freeze()
    learn.lr_find()
    learn.recorder.plot(skip_start=15)
    plt.show()

    learn.fit_one_cycle(1, 1e-02)
    learn.save('rest_head_pretrained')

    learn.unfreeze()
    learn.fit_one_cycle(10, 1e-03, moms=(0.8, 0.7))
    learn.save('rest_lm_fine_tuned')
    learn.save_encoder('rest_enc_fine_tuned')

if __name__ == "__main__":
    run()