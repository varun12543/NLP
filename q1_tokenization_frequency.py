import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('punkt_tab')   # Add this line
nltk.download('stopwords')

with open("sample.txt", "r", encoding="utf-8") as f:
    text = f.read()

words = nltk.word_tokenize(text)
stop_words = set(stopwords.words("english"))
filtered = [w.lower() for w in words if w.isalpha() and w.lower() not in stop_words]

fd = FreqDist(filtered)
print("Top 20 words:", fd.most_common(20))

fd.plot(20)
plt.show()
