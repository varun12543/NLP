import os
import nltk
from nltk.corpus import stopwords
from gensim import corpora, models, similarities

nltk.download('punkt')
nltk.download('stopwords')

folder = "docs"
docs = [open(os.path.join(folder, f), encoding="utf-8").read() for f in os.listdir(folder)]
stop_words = set(stopwords.words("english"))

texts = [[w for w in nltk.word_tokenize(d.lower()) if w.isalpha() and w not in stop_words] for d in docs]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(t) for t in texts]

tfidf = models.TfidfModel(corpus)
index = similarities.MatrixSimilarity(tfidf[corpus])

for i, sims in enumerate(index):
    print(f"Doc {i} similarities: {list(enumerate(sims))}")