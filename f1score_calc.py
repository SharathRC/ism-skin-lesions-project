import numpy as np

def f1score_calc(Yt,Yp):
    Yt = Yt.reshape(-1,1)
    Yp = Yp.reshape(-1,1)
    tp = sum(sum( [(Yt+Yp)==2] ))
    fp = sum(sum( [(2*Yp+Yt)==2] ))
    fn = sum(sum( [(Yp + 2*Yt)==2] ))
    tn = sum(sum( Yt == Yp)) - tp
        
    prec = tp/(tp+fp)
    rec = tp/(tp+fn)
    F1 = (2*prec*rec)/(prec+rec)
    
    print('tp: {}, tn:{}, fp: {}, fn: {}, prec: {}, rec: {}'.format(tp, tn, fp, fn, prec, rec))

    return F1