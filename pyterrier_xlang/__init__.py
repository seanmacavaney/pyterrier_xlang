from pyterrier_xlang.preprocess import *

_PREPROCESSORS = {
  'fa': spacy_tokeniser,
  'ru': spacy_preprocessor,
  'zh': jieba_preprocessor
}


def preprocess(lang, *args, **kwargs):
  return _PREPROCESSORS[lang](*args, **kwargs)
