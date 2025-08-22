import nltk
from nltk.tokenize import TextTilingTokenizer


for pkg in ["punkt", "punkt_tab"]:
    nltk.download(pkg, quiet=True)

with open("long_doc.txt", "r", encoding="utf-8") as f:
    data = f.read()


ttt = TextTilingTokenizer(w=10, k=5, smoothing_width=2)

segments = ttt.tokenize(data)

for i, seg in enumerate(segments):
    print(f"\nSegment {i+1} ({len(seg.split())} words):\n{seg}\n")
