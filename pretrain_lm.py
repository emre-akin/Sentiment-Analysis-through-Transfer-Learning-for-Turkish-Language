import pandas as pd
from fastai import *
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

    #LM Training
    PathCsv = Path('Path to your project folder')
    
    tokenizer = Tokenizer(lang='xx', n_cpus=4)
    data_lm_full = (TextList.from_csv(PathCsv, csv_name='fulltrain.csv', cols=0, processor=[TokenizeProcessor(tokenizer=tokenizer), NumericalizeProcessor(max_vocab=60000)])
                    #Inputs: all the text files in path
                    .split_from_df(col=1)
                    #We may have other temp folders that contain text files so we only keep what's in train and test
                    .label_for_lm()
                    #We want to do a language model so we label accordingly
                    .databunch(bs=32))
    
    data_lm_full.save('full_lm')
    
    
    data_lm_full = TextLMDataBunch.load(PathCsv, 'full_lm', bs=32)
    print(len(data_lm_full.train_ds.vocab.itos))
    data_lm_full.show_batch()

    learn = language_model_learner(data_lm_full, AWD_LSTM, drop_mult=0.3, callback_fns=ShowGraph)
    learn.lr_find()
    learn.recorder.plot(skip_start=0)
    plt.show()
    
    learn.fit_one_cycle(10, 1e-01, moms=(0.8,0.7))
    learn.save('model-full-v2')

    data_lm_full = TextLMDataBunch.load(PathCsv, 'full_lm', bs=32)
    print(len(data_lm_full.train_ds.vocab.itos))
    learn = language_model_learner(data_lm_full, AWD_LSTM)
    learn.load_pretrained(PathCsv 'models/model-full-v2.pth', PathCsv 'full_lm/itos.pkl')

    Text = 'Bu köyün özellikleri arasında '
    N_WORDS = 20
    N_SENTENCES = 2

    print("\n".join(learn.predict(Text, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))
    


if __name__ == "__main__":
    run()