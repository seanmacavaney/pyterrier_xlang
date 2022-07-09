import pyterrier as pt
pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])
import pyterrier_xlang
from pyterrier.measures import *

pre = pyterrier_xlang.preprocess('zh')
bm25 = pt.TerrierRetrieve('./hc4.zh.terrier', wmodel='BM25')
rm3 = pt.rewrite.RM3('./hc4.zh.terrier')

dataset = pt.get_dataset('irds:hc4/zh/test')

print(pt.Experiment(
  [pre >> bm25, pre >> bm25 >> rm3 >> bm25],
  dataset.get_topics('mt_title', tokenise_query=False),
  dataset.get_qrels(),
  [nDCG@100, AP@100, R@1000, Judged@10]
))
