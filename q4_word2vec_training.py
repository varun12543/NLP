import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec

for pkg in ["punkt", "punkt_tab", "stopwords"]:
    nltk.download(pkg, quiet=True)

with open("corpus.txt", "r", encoding="utf-8") as f:
    corpus = f.read()

stop_words = set(stopwords.words("english"))


raw_sentences = nltk.sent_tokenize(corpus)
sentences = [
    [w.lower() for w in nltk.word_tokenize(sentence) if w.isalpha() and w.lower() not in stop_words]
    for sentence in raw_sentences
]


model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, sg=0)


if "computer" in model.wv:
    print(model.wv.most_similar("computer", topn=10))
else:
    print("The word 'computer' is not in the vocabulary.")
