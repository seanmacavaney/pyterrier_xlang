import pyterrier as pt ; pt.init() ; from pyterrier_pisa import PisaIndex ; from pyterrier.measures import * ; import pyterrier_pisa
import pyterrier_xlang
import itertools
import pandas as pd
from pandarallel import pandarallel
#pandarallel.initialize(verbose=0)

dataset = pt.get_dataset('irds:hc4/zh/test')

topics = dataset.get_topics('ht_title', tokenise_query=False)
qrels = dataset.get_qrels()

for stem in [True]:
  proc = pyterrier_xlang.preprocess.zh()
  #x = next(iter(dataset.get_corpus_iter()))
  #res = proc(pd.DataFrame([x]))
  #import pdb; pdb.set_trace()
  #idx = PisaIndex(f'/home/sean/data/indices/hc4_zh.pisa', stemmer='none', pretokenised=True, text_field=['title', 'text'])
  idx = PisaIndex(f'/home/sean/data/indices/hc4_zh.stem-{stem}.pisa.old', stemmer='none', pretokenised=True, text_field=['title', 'text'])
  if not idx.built():
    (proc >> idx).index(dataset.get_corpus_iter(), batch_size=50000)
  #print(stem, norm, 'mt', pt.Experiment([proc >> idx.bm25()], dataset.get_topics('mt_title', tokenise_query=False), dataset.get_qrels(), [nDCG@100, AP@100, R@1000, Judged@10]))
  print(stem, pt.Experiment([proc >> idx.dph()], topics, qrels, [nDCG@100, AP@100, R@1000, Judged@10]))
