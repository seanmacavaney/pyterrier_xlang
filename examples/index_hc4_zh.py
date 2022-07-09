import pyterrier as pt
pt.init()
import pyterrier_xlang

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=False, nb_workers=15)

pre = pyterrier_xlang.preprocess('zh')
indexer = pt.IterDictIndexer('./hc4.zh.terrier', meta={"docno": 36})
indexer.setProperty("tokeniser", "UTFTokeniser")
indexer.setProperty("termpipelines", "")

(pre >> indexer).index(pt.get_dataset('irds:hc4/zh').get_corpus_iter(), batch_size=100000)
