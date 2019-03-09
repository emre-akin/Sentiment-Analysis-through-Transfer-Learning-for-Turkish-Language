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
    
    #data_lm = load_data('D:/Emre/source/data_equal/', 'data_lm_shop2_fn.pkl', bs=32)
    #print(data_lm)
    #tokenizer = Tokenizer(lang='tr')
    #data_clas = TextClasDataBunch.from_csv('D:/Emre/source/data_equal/' ,vocab=data_lm.vocab ,csv_name='shop2_full_clas.csv', tokenizer=tokenizer)
    #data_clas.save('shop2_data_clas.pkl')
    data_clas = load_data('D:/Emre/source/data_equal/', 'shop2_data_clas.pkl', bs=32)

    #data_clas = TextClasDataBunch.from_csv('D:/Emre/source/data_equal/',csv_name='shop2_full_clas.csv', bs=32)
    learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
    learn.load_encoder('shop2_enc_fine_tuned')
    f1_label1 = Fbeta_binary(1,clas = 0)
    f1_label0 = Fbeta_binary(1,clas = 1)
    learn.metrics=[accuracy, f1_label1,f1_label0]
    learn.freeze()
    
    learn.lr_find()
    learn.recorder.plot()
    plt.show()
    print(learn.model)
    learn.fit_one_cycle(1, 1.45e-01, moms=(0.8,0.7))
    #learn.save('shop2_first')
    #learn.load('shop2_first')

    learn.freeze_to(-2)
    learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))
    #learn.save('shop2_second')
    #learn.load('shop2_second')

    learn.freeze_to(-3)
    learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))
    #learn.save('shop2_third')  
    #learn.load('shop2_third')

    learn.unfreeze()
    learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))
    print(learn.predict("Güzel ürün tavsiye ederim."))
    print(learn.predict("Kötü"))
    print(learn.predict("Rezalet ötesi"))
    
    
if __name__ == "__main__":
    run()