import string
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


def fa(normalise=True, stem=True, remove_stops=True, remove_punct=True):
  try:
    import hazm
  except ImportError as e:
    raise ImportError("pip install hazm to perform fa preprocessing", e)
  stemmer = None
  if stem == 'lemma':
    stemmer = hazm.Lemmatizer().lemmatize
  elif stem:
    stemmer = hazm.Stemmer().stem
  term_filter = None
  if remove_stops or remove_punct:
    term_filter = lambda t: True
    if remove_stops:
      stops = set(hazm.stopwords_list())
      def filter_stops(f):
        return lambda t: f(t) and t not in stops
      term_filter = filter_stops(term_filter)
    if remove_punct:
      def filter_punct(f):
        return lambda t: f(t) and t not in string.punctuation
      term_filter = filter_punct(term_filter)
  return Preprocessor(hazm.word_tokenize, stemmer=stemmer, preprocessor=hazm.Normalizer().normalize if normalise else None, term_filter=term_filter)


def spacy_preprocessor(model, supports_stem=True, stops=None):
  def wrapped(stem=True, remove_punct=True, remove_stops=True):
    try:
      import spacy
    except ImportError as e:
      raise ImportError("pip install spacy required", e)
    try:
      nlp = spacy.load(model, disable=['ner', 'parser', 'tok2vec'])
    except OSError as e:
      raise RuntimeError(f"Problem loading model {model} (do you need to do python -m spacy download {model}?)", e)
    if supports_stem and stem:
      stemmer = lambda t: t.lemma_
    else:
      stemmer = lambda t: t.norm_
    term_filter = lambda t: True
    if remove_stops:
      if stops is not None:
        def filter_stops(f):
          return lambda t: f(t) and not t.is_stop
      else:
        def filter_stops(f):
          return lambda t: f(t) and str(t) not in stops
      term_filter = filter_stops(term_filter)
    if remove_punct:
      def filter_punct(f):
        return lambda t: f(t) and not t.is_punct
      term_filter = filter_punct(term_filter)
    return Preprocessor(nlp, stemmer=stemmer, term_filter=term_filter)
  return wrapped

ru = spacy_preprocessor('ru_core_news_sm')
zh = spacy_preprocessor('zh_core_web_sm', supports_stem=False, stops=set("按 按照 俺 俺们 阿 别 别人 别处 别是 别的 别管 别说 不 不仅 不但 不光 不单 不只 不外乎 不如 不妨 不尽 不尽然 不得 不怕 不惟 不成 不拘 不料 不是 不比 不然 不特 不独 不管 不至于 不若 不论 不过 不问 比方 比如 比及 比 本身 本着 本地 本人 本 巴巴 巴 并 并且 非彼 彼时 彼此 便于 把 边 鄙人 罢了 被 般的 此间 此次 此时 此外 此处 此地 此 才 才能 朝 朝着 从 从此 从而 除非 除此之外 除开 除外 除了 除 诚然 诚如 出来 出于 曾 趁着 趁 处在 乘 冲 等等 等到 等 第 当着 当然 当地 当 多 多么 多少 对 对于 对待 对方 对比 得 得了 打 打从 的 的确 的话 但 但凡 但是 大家 大 地 待 都 到 叮咚 而言 而是 而已 而外 而后 而况 而且 而 尔尔 尔后 尔 二来 非独 非特 非徒 非但 否则 反过来说 反过来 反而 反之 分别 凡是 凡 个 个别 固然 故 故此 故而 果然 果真 各 各个 各位 各种 各自 关于具体地说 归齐 归 根据 管 赶 跟 过 该 给 光是 或者 或曰 或是 或则 或 何 何以 何况 何处 何时 还要 还有 还是 还 后者 很 换言之 换句话说 好 后 和 即 即令 即使 即便 即如 即或 即若 继而 继后 继之 既然 既是 既往 既 尽管如此 尽管 尽 就要 就算 就是说 就是了 就是 就 据 据此 接着 经 经过 结果 及 及其 及至 加以 加之 例如 介于 几时 几 截至 极了 简言之 竟而 紧接着 距 较之 较 进而 鉴于 基于 具体说来 兼之 借傥然 今 叫 将 可 可以 可是 可见 开始 开外 况且 靠 看 来说 来自 来着 来 两者 临 类如 论 赖以 连 连同 离 莫若 莫如 莫不然 假使 假如 假若 某 某个 某些 某某 漫说 没奈何 每当 每 慢说 冒 哪个 哪些 哪儿 哪天 哪年 哪怕 哪样 哪边 哪里 那里 那边 那般 那样 那时 那儿 那会儿 那些 那么样 那么些 那么 那个 那 乃 乃至 乃至于 宁肯 宁愿 宁可 宁 能 能否 你 你们 您 拿 难道说 内 哪 凭借 凭 旁人 譬如 譬喻 且 且不说 且说 其 其一 其中 其二 其他 其余 其它 其次 前后 前此 前者 起见 起 全部 全体 恰恰相反 岂但 却 去 若非 若果 若是 若夫 若 另 另一方面 另外 另悉 如若 如此 如果 如是 如同 如其 如何 如下 如上所述 如上 如 然则 然后 然而 任 任何 任凭 仍 仍旧 人家 人们 人 让 甚至于 甚至 甚而 甚或 甚么 甚且 什么 什么样 上 上下 虽说 虽然 虽则 虽 孰知 孰料 始而 所 所以 所在 所幸 所有 是 是以 是的 设使 设或 设若 谁 谁人 谁料 谁知 随着 随时 随后 随 顺着 顺 受到 使得 使 似的 尚且 庶几 庶乎 时候 省得 说来 首先 倘 倘使 倘或 倘然 倘若 同 同时 他 他人 他们们 她们 她 它们 它 替代 替 通过 腾 这里 这边 这般 这次 这样 这时 这就是说 这儿 这会儿 这些 这么点儿 这么样 这么些 这么 这个 这一来 这 正是 正巧 正如 正值 万一 为 为了 为什么 为何 为止 为此 为着 无论 无宁 无 我们 我 往 望 惟其 唯有 下 向着 向使 向 先不先 相对而言 许多 像 小 些 一 一些 一何 一切 一则 一方面 一旦 一来 一样 一般 一转眼 由此可见 由此 由是 由于 由 用来 因而 因着 因此 因了 因为 因 要是 要么 要不然 要不是 要不 要 与 与其 与其说 与否 与此同时 以 以上 以为 以便 以免 以及 以故 以期 以来 以至 以至于 以致 己 已 已矣 有 有些 有关 有及 有时 有的 沿 沿着 于 于是 于是乎 云云 云尔 依照 依据 依 余外 也罢 也好 也 又及 又 抑或 犹自 犹且 用 越是 只当 只怕 只是 只有 只消 只要 只限 再 再其次 再则 再有 再者 再者说 再说 自身 自打 自己 自家 自后 自各儿 自从 自个儿 自 怎样 怎奈 怎么样 怎么办 怎么 怎 至若 至今 至于 至 纵然 纵使 纵令 纵 之 之一 之所以 之类 着呢 着 眨眼 总而言之 总的说来 总的来说 总的来看 总之 在于 在下 在 诸 诸位 诸如 咱们 咱 作为 只 最 照着 照 直到 综上所述 贼死 逐步 遵照 遵循 针对 致 者 则甚 则 咳 哇 哈 哈哈 哉 哎 哎呀 哎哟 哗 哟 哦 哩 矣哉 矣乎 矣 焉 毋宁 欤 嘿嘿 嘿 嘻 嘛 嘘 嘎登 嘎 嗳 嗯 嗬 嗡嗡 嗡 喽 喔唷 喏 喂 啷当 啪达 啦 啥 啐 啊 唉 哼唷 哼 咧 咦 咚 咋 呼哧 呸 呵呵 呵 呢 呜呼 呜 呗 呕 呃 呀 吱 吧哒 吧 吗 吓 兮 儿 亦 了 乎".split()))

"""
def zh(remove_punct=True):
  term_filter = None
  if remove_punct:
    term_filter = lambda t: t not in string.punctuation
  return Preprocessor(lambda x: list(x), term_filter=term_filter)
"""
