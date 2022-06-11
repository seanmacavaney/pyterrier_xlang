import pyterrier as pt ; pt.init() ; from pyterrier_pisa import PisaIndex ; from pyterrier.measures import * ; import pyterrier_pisa
import pyterrier_xlang
import itertools
from pandarallel import pandarallel
#pandarallel.initialize(verbose=0)

dataset = pt.get_dataset('irds:hc4/fa/test')

topics = dataset.get_topics('ht_title', tokenise_query=False)
qrels = dataset.get_qrels()

for stem, norm in [('lemma', True)]:
  pre = pyterrier_xlang.preprocess.fa(stem=stem, normalise=norm)
  sent = pyterrier_xlang.hazm.sents(length=4, stride=2)
  idx = PisaIndex(f'/home/sean/data/indices/hc4_fa.stem-{stem}.norm-{norm}.sents.pisa', stemmer='none', pretokenised=True, text_field=['text'])
  if not idx.built():
    (sent >> pre >> idx).index(dataset.get_corpus_iter(), batch_size=50000)
  res = (pre >> idx.bm25(num_results=10000))(topics)
  import pdb; pdb.set_trace()
  #print(stem, norm, 'mt', pt.Experiment([proc >> idx.bm25()], dataset.get_topics('mt_title', tokenise_query=False), dataset.get_qrels(), [nDCG@100, AP@100, R@1000, Judged@10]))
  print(stem, norm, pt.Experiment([pre >> idx.bm25(num_results=10000) >> pt.text.max_passage()], topics, qrels, [nDCG@100, AP@100, R@1000, Judged@10]))
