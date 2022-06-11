import pyterrier as pt ; pt.init() ; from pyterrier_pisa import PisaIndex ; from pyterrier.measures import * ; import pyterrier_pisa
import pyterrier_xlang
import itertools
from pandarallel import pandarallel
pandarallel.initialize(verbose=0)

dataset = pt.get_dataset('irds:hc4/ru/train')

topics = dataset.get_topics('ht_title', tokenise_query=False)
qrels = dataset.get_qrels()

for stem in [True]:
  proc = pyterrier_xlang.preprocess.ru(stem=stem)
  idx = PisaIndex(f'/home/sean/data/indices/hc4_ru.stem-{stem}.pisa', stemmer='none', pretokenised=True, text_field=['title', 'text'])
  if not idx.built():
    (proc >> idx).index(dataset.get_corpus_iter(), batch_size=10000)
  #print(stem, norm, 'mt', pt.Experiment([proc >> idx.bm25()], dataset.get_topics('mt_title', tokenise_query=False), dataset.get_qrels(), [nDCG@100, AP@100, R@1000, Judged@10]))
  #print(stem, norm, pt.Experiment([proc >> idx.bm25()], topics, qrels, [nDCG@100, AP@100, R@1000, Judged@10]))
