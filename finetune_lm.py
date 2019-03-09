import pandas as pd
from fastai.text import *
import matplotlib.pyplot as plt

@dataclass
class Fbeta_binary(Callback):
    "Computes the fbeta between preds and targets for single-label classification"
    beta2: int = 2
    eps: float = 1e-9
    clas:int=1
    
    def on_epoch_begin(self, **kwargs):
        self.TP = 0
        self.total_y_pred = 0   
        self.total_y_true = 0
    
    def on_batch_end(self, last_output, last_target, **kwargs):
        y_pred = last_output.argmax(dim=1)
        y_true = last_target.float()
        
        self.TP += ((y_pred==self.clas) * (y_true==self.clas)).float().sum()
        self.total_y_pred += (y_pred==self.clas).float().sum()
        self.total_y_true += (y_true==self.clas).float().sum()
    
    def on_epoch_end(self, **kwargs):
        beta2=self.beta2**2
        prec = self.TP/(self.total_y_pred+self.eps)
        rec = self.TP/(self.total_y_true+self.eps)       
        res = (prec*rec)/(prec*beta2+rec+self.eps)*(1+beta2)
        self.metric = res

def freeze_support():
    ''' Check whether this is a fake forked process in a frozen executable. If so then run code specified by commandline and exit. ''' 
    if sys.platform == 'win32' and getattr(sys, 'frozen', False):
        from multiprocessing.forking import freeze_support
        freeze_support()
def run():
    freeze_support()
    print('loop')
    
    tokenizer = Tokenizer(lang='tr')
    data_lm = TextLMDataBunch.from_csv('D:/Emre/source/data_equal/', csv_name='shop2_full.csv',text_cols=0, tokenizer=tokenizer)
    data_lm.save('data_lm_shop2_fn.pkl')
    
    data_lm = load_data('D:/Emre/source/data_equal/', 'data_lm_shop2_fn.pkl', bs=32)
    learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3 )
    learn.load_pretrained('D:/Emre/source/data/wiki/tr/models/model-full-v2.pth','D:/Emre/source/data/wiki/tr/full_lm_60k/itos.pkl')
    #learn.freeze()
    learn.lr_find()
    learn.recorder.plot(skip_start=15)
    plt.show()

    learn.fit_one_cycle(1, 1e-02)
    learn.save('shop2_head_pretrained')

    learn.unfreeze()
    learn.fit_one_cycle(10, 1e-03, moms=(0.8, 0.7))
    learn.save('shop2_lm_fine_tuned')
    learn.save_encoder('shop2_enc_fine_tuned')

if __name__ == "__main__":
    run()