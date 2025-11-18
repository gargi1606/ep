``` Python
##EXP 7
import pandas as pd

df=pd.read_csv('/content/sentimentdataset (1).csv')
df



# Install libraries
!pip install gensim pyLDAvis

import pandas as pd
import nltk, gensim, pyLDAvis.gensim
from gensim import corpora
from nltk.corpus import stopwords


# Load data
texts = df['Text'].astype(str).tolist()



# Preprocessing
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

stop_words = set(stopwords.words("english"))

def clean(t):
    tokens = nltk.word_tokenize(t.lower())
    return [w for w in tokens if w.isalpha() and w not in stop_words]

processed = [clean(t) for t in texts]


# Dictionary & Corpus
dictionary = corpora.Dictionary(processed)
corpus = [dictionary.doc2bow(text) for text in processed]



# LDA Model
lda = gensim.models.LdaModel(corpus, id2word=dictionary, num_topics=5, passes=5)



# Print Topics
for i, topic in lda.print_topics():
    print(f"Topic {i}:", topic)




# Visualization
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
pyLDAvis.save_html(vis, "lda.html")

print("✔ LDA topics saved as lda.html")

from google.colab import files
files.download("/content/lda.html")
```

``` Python
#EXP 5
import nltk
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Download tokenizer
nltk.download('punkt')

# Load dataset
df = pd.read_csv('/content/social_media_behavior_dataset.csv')

# Keep ONLY 3 classes
sentiment_df = df[df['Sentiment'].isin(["Positive", "Negative", "Neutral"])].copy()

# Tokenization
sentiment_df['Tokens'] = sentiment_df["Post Content"].astype(str).apply(nltk.word_tokenize)

print("\nSample Tokenization:")
for i in range(5):
    print("Text:", sentiment_df['Post Content'].iloc[i])
    print("Tokens:", sentiment_df['Tokens'].iloc[i])
    print()

# TRUE labels
y_true = sentiment_df["Sentiment"]

# PREDICTED labels (dummy prediction: predict same label)
# If you want model predictions, plug in your model here.
y_pred = sentiment_df["Sentiment"]

# Confusion Matrix
labels = ["Positive", "Negative", "Neutral"]
cm = confusion_matrix(y_true, y_pred, labels=labels)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, labels=labels))


df

import nltk
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report
import seaborn as sns


nltk.download('punkt')

df=pd.read_csv('/content/social_media_behavior_dataset.csv')

#keep only three classes
sentiment_df = df[df['Sentiment'].isin(["Positive", "Negative", "Neutral"])].copy()

#tokenziation
sentiment_df['Tokens'] = sentiment_df["Post Content"].astype(str).apply(nltk.word_tokenize)
```
