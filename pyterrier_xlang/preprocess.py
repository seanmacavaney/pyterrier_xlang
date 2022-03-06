import pyterrier as pt

class Preprocessor(pt.transformer.TransformerBase):
  def __init__(self, tokeniser, stemmer=None, preprocessor=None, text_field='text'):
    self.preprocessor = preprocessor
    self.tokeniser = tokeniser
    self.stemmer = stemmer
    self.text_field = [text_field] if isinstance(text_field, str) else text_field

  def transform(self, df):
    if 'query' in df.columns:
      if hasattr(df, 'parallel_apply'):
        df = df.assign(query=df['query'].parallel_apply(self.process_text))
      else:
        df = df.assign(query=df['query'].apply(self.process_text))
    if hasattr(df, 'parallel_apply'):
      df = df.assign(**{f: df[f].parallel_apply(self.process_text) for f in self.text_field if f in df.columns})
    else:
      df = df.assign(**{f: df[f].apply(self.process_text) for f in self.text_field if f in df.columns})
    return df

  def process_text(self, s):
    if self.preprocessor:
      s = self.preprocessor(s)
    toks = self.tokeniser(s)
    if self.stemmer:
      toks = (self.stemmer(t) for t in toks)
    return ' '.join(toks)


def fa(normalise=True, stem=True):
  try:
    import hazm
  except ImportError as e:
    raise ImportError("pip install hazm to perform fa preprocessing", e)
  if stem == 'lemma':
    stemmer = hazm.Lemmatizer().lemmatize
  elif stem:
    stemmer = hazm.Stemmer().stem
  else:
    stemmer = None
  return Preprocessor(hazm.word_tokenize, stemmer=stemmer, preprocessor=hazm.Normalizer().normalize if normalise else None)
