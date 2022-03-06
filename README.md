# pyterrier_xlang

PyTerrier components for cross-language retrieval.

## Preprocessing

`pyterrier_xlang.preprocess` provides tools for performing text pre-processing, such as tokenisation and stemming.

**pyterrier_xlang.preprocess.fa: Persian/Farsi**

Performs noramlisation, tokenisation, amd stemming for queries and documents via the [hazm](https://github.com/sobhe/hazm) package.

```
pyterrier_xlang.preprocess.fa(normalise=True, stem=True)
```

 - `normalise` (bool, default=True): Perform text normalisation via `hazm.Normalize()`, e.g., "" to «», diacritic removal, persian numeral replacement, etc.
 - `stem` (bool/str, default=True): If True, perform stemming via `hazm.Stemmer()`. If "lemma", perform lemmatisation via `hazm.Lemmatizer()`. If `False`, do not perform any stemming.

Examples:

```python
>>> fa_pre = pyterrier_xlang.preprocess.fa()
# 1: Apply preprocessing to query or a set of queries:
>>> fa.search('اعتراض فرانسه به مالیات سوخت')
# qid                     query
#   1  اعتراض فرانسه به مال سوخ
>>> dataset = pt.get_dataset('irds:hc4/fa/train')
>>> topics = dataset.get_topics('mt_title', tokenise_query=False)
>>> fa_pre(topics)
#   qid                                     query
#  1008                  اعتراض فرانسه به مال سوخ
#  1013                حداکثر ۸ سقوط تاثیر بوئینگ
#  1020       تأثیر سیاس خروج از توافق نامه پاریس
#  1021                      آمازون رقاب ضد رقابت
#  1022                   پناهندگ روهینگیا بنگلاد
#  1023        تیربار خمپاره تروریست در روستا حلب
#  1026                      قوانین تجاوز جنس هند
#  1027  کمپین جنگ مواد مخدر ایال متحده و فیلیپین

# 2: Index using pre-processor:
>>> from pyterrier_pisa import PisaIndex
>>> idx = PisaIndex('hc4_fa.pisa', stemmer='none', pretokenised=True, text_field=['title', 'text'])
>>> (fa_pre >> idx).index(dataset.get_corpus_iter())
# (index built at hc4_fa.pisa)

# 3: BM25 retrieval and evaluation using pre-processor
>>> retr_pipeline = fa_pre >> idx.bm25()
>>> retr_pipeline(dataset.get_topics('mt_title', tokenise_query=False))
#    qid                                 docno  rank      score                                query
#   1022  025471d0-e2c5-4f90-80c5-923ce2e99005     1  30.891193              پناهندگ روهینگیا بنگلاد
#   1022  60918139-d737-46bd-8f5d-600de41e7887     2  30.627840              پناهندگ روهینگیا بنگلاد
..   ...                                   ...   ...        ...                                  ...
#   1020  74f7301f-06fa-4a78-bb4b-1538f69803ab   999  11.609644  تأثیر سیاس خروج از توافق نامه پاریس
#   1020  74bb59de-e63b-44b7-9493-bea3f87d7a0c  1000  11.609144  تأثیر سیاس خروج از توافق نامه پاریس
# [8000 rows x 5 columns]
>>> from pyterrier.measures import *
>>> pt.Experiment([retr_pipeline], topics, dataset.get_qrels(), [nDCG@100, AP@100, R@1000, Judged@10])
# nDCG@100    AP@100   R@1000  Judged@10
# 0.495567  0.373508  0.96875     0.3125
```
