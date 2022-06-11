import more_itertools
import pandas as pd
import pyterrier as pt

class Preprocessor(pt.transformer.TransformerBase):
  def __init__(self, tokeniser, stemmer=None, preprocessor=None, term_filter=None, text_fields=['title', 'text', 'body'], push_query=True):
    self.preprocessor = preprocessor
    self.tokeniser = tokeniser
    self.term_filter = term_filter
    self.stemmer = stemmer
    self.text_fields = [text_fields] if isinstance(text_fields, str) else text_fields
    self.push_query = push_query

  def transform(self, df):
    if self.push_query and 'query' in df.columns:
      pt.model.push_queries(df)
    if hasattr(df, 'parallel_apply'):
      df = df.assign(**{f: df[f].parallel_apply(self.process_text) for f in self.text_fields + ['query'] if f in df.columns})
    else:
      df = df.assign(**{f: df[f].apply(self.process_text) for f in self.text_fields + ['query'] if f in df.columns})
    return df

  def process_text(self, s):
    if self.preprocessor:
      s = self.preprocessor(s)
    toks = self.tokeniser(s)
    if self.term_filter:
      toks = filter(self.term_filter, toks)
    if self.stemmer:
      toks = map(self.stemmer, toks)
    return ' '.join(toks)


class SentencePassager(pt.transformer.TransformerBase):
  def __init__(self, sent_tokenize, length=4, stride=2, text_field='text', prepend_field='title'):
    self.sent_tokenize = sent_tokenize
    self.length = length
    self.stride = stride
    assert self.stride <= self.length
    self.text_field = text_field
    self.prepend_field = prepend_field

  def transform(self, df):
    applied_df = df.apply(self.process_text, axis='columns', result_type='expand').rename(columns={0: 'docno', 1: self.text_field})
    df = pd.concat([df.drop(columns=['docno', self.text_field, self.prepend_field], errors='ignore'), applied_df], axis='columns')
    return df.explode(['docno', self.text_field], ignore_index=True)

  def process_text(self, row):
    sents = self.sent_tokenize(row[self.text_field])
    if self.length != 1 and self.stride != 1:
      it = enumerate(' '.join(w) for w in more_itertools.windowed(sents, self.length, fillvalue='', step=self.stride))
    else:
      it = enumerate(sents)
    docnos = []
    texts = []
    for i, passage in it:
      docnos.append(f'{row["docno"]}%p{i}')
      if self.prepend_field and self.prepend_field in row:
        passage = f'{row[self.prepend_field]} {passage}'
      texts.append(passage)
    return docnos, texts
