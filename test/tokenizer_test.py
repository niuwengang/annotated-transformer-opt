import spacy

# 1. 加载两个分词器
spacy_de = spacy.load("de_core_news_sm")
spacy_en = spacy.load("en_core_web_sm")

# 2. 分词
de_text = "Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche."
en_text = "Two young white men are outside near many bushes."

de_tokens = []
for tok in spacy_de.tokenizer(de_text):
    de_tokens.append(tok.text)

en_tokens = []
for tok in spacy_en.tokenizer(en_text):
    en_tokens.append(tok.text)

print("德语分词:", de_tokens)
print("英语分词:", en_tokens)
