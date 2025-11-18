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

```Python
# ===========================
# EXPERIMENT 3
# INFO + WORDCLOUD
# ===========================

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS



# Load EXP 2 output manually (paste sentiment_df or load from CSV)
df = pd.read_csv("/content/social_media_behavior_dataset.csv")

df.info()

df

sentiment_df=df[['Post Content','Sentiment']]
sentiment_df

print("\nSentiment Distribution:")
print(sentiment_df['Sentiment'].value_counts(normalize=True))

# Combine all comments into one string
comment_text = " ".join(sentiment_df["Post Content"])

# --- WORDCLOUD ---
word_cloud = WordCloud(
    background_color="white",
    stopwords=ENGLISH_STOP_WORDS,
    width=900,
    height=300
).generate(comment_text)

plt.rcParams["figure.figsize"] = (13, 10)
plt.imshow(word_cloud, interpolation="bilinear")
plt.axis("off")
plt.show()
```

```Python
#exp 4
import pandas as pd
import 6matplotlib.pyplot as plt
from c6ollections import Counter

df=pd.read_csv('/content/sentimentdataset.csv')
df

def word_frequency(column):
  words=" ".join(column.astype(str)).split()
  return Counter(words)

def common_hashtags(column):
  hshtags=[]
  for text in column:
    if isinstance(text,str):
      hshtags.extend(text.split()) # Changed 'hashtags' to 'hshtags' and added '()'
  return Counter(hshtags) # Changed 'hashtags' to 'hshtags'

def top_users(df,n=5):
  return(df['User'].value_counts().head(n))

#Run function
word_fre=word_frequency(df['Text'])
calculated_common_hashtags=common_hashtags(df['Hashtags'])
top_users=top_users(df)

#print results
print("Top 10 word frequency:\n",word_freq.most_common(10))
print("\nTop 10 common hashtags:\n",calculated_common_hashtags.most_common(10))
print("\nTop 5 users:\n",top_users)

#plot 5 top hashtags
plt.figure(figsize=(10,5))
plt.bar(*zip(*calculated_common_hashtags.most_common(5)))
plt.xlabel('Hashtags')
plt.ylabel('Frequency')
plt.title('Top 5 Hashtags')
plt.show()
```

```Python
#expt 2
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

df=pd.read_csv('/content/sentimentdataset.csv')

df

df.info()

sentiment_df=df[['Text','Sentiment']]
sentiment_df

print("Sentiment Distribution")
print(sentiment_df['Sentiment'].value_counts())
#combine all comments into  word strings
comment_text=sentiment_df['Text'].str.cat(sep=' ')
#wordcloud
word_cloud=WordCloud(
    background_color='white',
    stopwords=set(ENGLISH_STOP_WORDS),
    max_words=100

).generate(comment_text)



#plot wordcloud
plt.rcParams['figure.figsize']=(13,10)
plt.imshow(word_cloud,interpolation='bilinear')
plt.axis('off')
plt.show()



Word freq,Hshtages,Top Users
```