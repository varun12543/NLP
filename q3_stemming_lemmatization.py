import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

with open("sample.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokens = nltk.word_tokenize(text)
words = [w for w in tokens if w.isalpha()][:15]

ps = PorterStemmer()
lm = WordNetLemmatizer()

print(f"{'Word':<15}{'Stemmed':<15}{'Lemmatized'}")
for w in words:
    print(f"{w:<15}{ps.stem(w):<15}{lm.lemmatize(w)}")