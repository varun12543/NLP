import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

with open("long_doc.txt", "r", encoding="utf-8") as f:
    data = f.read()

paragraphs = [p for p in data.split("\n") if p.strip()]
vectorizer = TfidfVectorizer(stop_words='english')
tfidf = vectorizer.fit_transform(paragraphs)
similarity = cosine_similarity(tfidf)

threshold = 0.3
print("\nBoundaries found:\n")
for i in range(len(similarity) - 1):
    if similarity[i, i + 1] < threshold:
        print(f"Between paragraph {i+1} and {i+2}")