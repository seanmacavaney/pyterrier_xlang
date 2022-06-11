import pyterrier as pt ; pt.init() ; from pyterrier_pisa import PisaIndex ; from pyterrier.measures import * ; import pyterrier_pisa
import pyterrier_xlang
import itertools
from pandarallel import pandarallel
#pandarallel.initialize(verbose=0)
import pyterrier_sbert

dataset = pt.get_dataset('irds:hc4/fa/test')

topics = dataset.get_topics('ht_description', tokenise_query=False)
qrels = dataset.get_qrels()

for stem, norm in [('lemma', True)]:
  pre = pyterrier_xlang.preprocess.fa(stem=stem, normalise=norm)
  sent = pyterrier_xlang.hazm.sents(length=4, stride=2)
  #idx = pyterrier_sbert.NumpyIndex(f'/home/sean/data/indices/hc4_fa.stem-{stem}.norm-{norm}.sents.distiluse-base-multilingual-cased-v2.np')
  idx = pyterrier_sbert.NumpyIndex(f'/home/sean/data/indices/hc4_fa.stem-{stem}.norm-{norm}.sents.distiluse-base-multilingual-cased-v2.np')
  sidx = PisaIndex(f'/home/sean/data/indices/hc4_fa.stem-{stem}.norm-{norm}.sents.pisa', stemmer='none', pretokenised=True, text_field=['text'])
  model = pyterrier_sbert.SBert('sentence-transformers/distiluse-base-multilingual-cased-v2')
  if not idx.index_path.exists():
    #(sent >> model >> idx).index(dataset.get_corpus_iter(), batch_size=1000)
    (model >> idx).index(dataset.get_corpus_iter(), batch_size=1000)
  #print(pt.Experiment([model >> idx >> pt.text.max_passage()], topics, qrels, [nDCG@100, AP@100, R@1000, Judged@10]))
  print(pt.Experiment([
    model >> idx >> pt.text.max_passage(),
    model >> pre >> sidx.bm25() >> idx >> pt.text.max_passage(),
    pre >> sidx.bm25() >> pt.text.max_passage(),
  ], topics, qrels, [nDCG@100, AP@100, R@1000, Judged@10, Compat]))
