#!/usr/bin/env python
# coding: utf-8

# In[5]:


import nltk
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from string import punctuation
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D, Convolution1D
from keras.models import Model, Sequential
from keras import initializers, regularizers, constraints, optimizers, layers


# In[6]:


df = pd.read_csv('C:/Users/dedsec/Downloads/flipkart.csv')


# In[7]:


df.head(10)


# In[8]:


df.columns


# In[9]:


df.info()


# In[10]:


df.isnull().sum()


# In[11]:


# Checking unique values in all features
l=['marketplace','product_category','verified_purchase','vine','helpful_votes', 'total_votes']

for i in l:
  print('The unique element in : ', i)
  print('-'*30)
  print(df[i].unique())
  print()


# In[12]:


# Frequency of values of different types in all features
l=['product_category','verified_purchase','vine','helpful_votes', 'total_votes']

for i in l:
  print('The unique element in : ', i)
  print('-'*30)
  print(df[i].value_counts())
  print()


# In[13]:


# Plots to give visuals about frequency of values of all types in different features
features = ['product_category','verified_purchase','vine','helpful_votes','total_votes']
for i in features:
  print('The count plot of  : ', i)
  print('-'*30)
  plt.figure(figsize=(10,5))
  sns.countplot(df[i])
  plt.show()
  print()


# In[14]:


# Value count of every rating type
df.star_rating.value_counts()


# In[15]:


# Visualizing the value count of every rating type
plt.figure()
plt.figure(figsize=(4,4),dpi=100)
sns.countplot(x='star_rating',data=df)


# In[16]:


# Percentage of every rating w.r.t all other ratings
print('-' * 20 + "Percentage ratings" + '-' * 20)
star_ratings = df['star_rating'].value_counts() / len(df) * 100
star_ratings


# In[17]:


# Visualization of percentage of every rarting type
plt.figure()
sns.barplot(star_ratings.index, star_ratings.values, order=star_ratings.index)
plt.ylabel("Percentage")
plt.xlabel('Rating')


# In[18]:


df = df[df['star_rating'] != 3]


# In[19]:


df['rating'] = df['star_rating'].apply(lambda x: 1 if x >= 4 else 0)
df['rating'].value_counts()


# In[20]:


# Value count of positive and negative classes
plt.figure()
sns.countplot(x = 'rating', data = df, palette='rainbow')
plt.ylabel("Ratings count")
plt.xlabel("Rating")


# In[21]:


df['reviews'] = df['review_body'] + " " + df['review_headline']
df.head(10)


# In[22]:


df_new = df[['rating','reviews']]
df_new.head(10)


# In[23]:


# Initializing WordNet and stopwords on nlp model
lemmatizer = WordNetLemmatizer()
nltk.download('stopwords')
nltk.download('wordnet')

df_new['reviews'] = df_new['reviews'].apply(lambda sentence: sentence.lower())
df_new.head(10)


# In[24]:


# Compiling emoji patterns using regular expressions
emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags 
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)


# In[25]:


# Replacing the decontracted words with their original form
def decontracting_words(sentence):
    # specific
    sentence = re.sub(r"won\'t", "will not", sentence)
    sentence = re.sub(r"can\'t", "can not", sentence)

    # general
    sentence = re.sub(r"n\'t", " not", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'s", " is", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'t", " not", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'m", " am", sentence)
    return sentence


# In[26]:


# Removing HTML, XML, emoticons and decontracting words
stopwrds = stopwords.words('english')
def remove_html_urls_emoticons(sentence):
    # Remove HTML, XML tags from data 
    sentence = BeautifulSoup(sentence, 'lxml').get_text()
    # Remove URLs from the data 
    sentence = re.sub("http\S+", "", sentence)
    # Decontract words
    sentence = decontracting_words(sentence)
    # Removing emoticons
    sentence = emoji_pattern.sub(' ', sentence)
    return sentence


# In[27]:


# Removing words with digits in them and punctuations
def remove_numeric_punctuations_stopwords(sentence):
    # Remove words with numbers in them from the data
    sentence = re.sub(r"\S*\d\S*", "", sentence)
    # remove punctuations, numbers
    sentence = re.sub(r"[^A-Za-z]", ' ', sentence)
    # Remove stopwords
    sentence = " ".join([word for word in sentence.split() if word not in stopwrds])
    return sentence


# In[28]:


df_new['reviews'] = df_new['reviews'].apply(remove_html_urls_emoticons)
df_new['reviews'] = df_new['reviews'].apply(remove_numeric_punctuations_stopwords)


# In[29]:


# Lemmatizing the words
df_new['reviews'] = df_new['reviews'].apply(lambda x: " ".join([lemmatizer.lemmatize(token) for token in x.split()]))


# In[30]:


# Making word cloud of most popular words
from wordcloud import WordCloud

for i in range (0,2):
  print('When star Rating is : ', i)
  print('-' * 50)
  print()
  text = df_new[df_new['rating'] == i]
  all_words = ' '.join([text for text in text.reviews])
  wordcloud = WordCloud(width= 1500, height= 800,
                              max_font_size = 170,
                              collocations = False).generate(all_words)
  plt.figure(figsize=(10,7))
  plt.imshow(wordcloud, interpolation='bilinear')
  plt.axis("off")
  plt.show()


# In[31]:


# Tokenizing the words and visually representing the frequency of most popular words used
from nltk import tokenize

token_space = tokenize.WhitespaceTokenizer()
def counter(text, column_text, quantity):
    all_words = ' '.join([text for text in text[column_text]])
    token_phrase = token_space.tokenize(all_words)
    frequency = nltk.FreqDist(token_phrase)
    data_frequency = pd.DataFrame({"Word": list(frequency.keys()),
                                   "Frequency": list(frequency.values())})
    data_frequency = data_frequency.nlargest(columns = "Frequency", n = quantity)
    plt.figure(figsize=(15,8))
    ax = sns.barplot(data = data_frequency, x = "Word", y = "Frequency", color = 'maroon')
    ax.set(ylabel = "Count")
    plt.xticks(rotation='vertical')
    plt.show()


for i in range(0,2):
  print('When star Rating is : ', i)
  print('_'*30)
  print()
  counter(df_new[df_new['rating'] == i], 'reviews', 20)
  print()


# In[32]:


X = df_new['reviews']
Y = df_new['rating']


# In[33]:


# Splitting data into train test set
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)


# In[34]:


print(x_train.shape)
print('-' * 50)
print(x_test.shape)
print('-' * 50)
print(y_train.shape)
print('-' * 50)
print(y_test.shape)
print('-' * 50)


# In[35]:


# Tokenizing the words using Tokenizer
# Using the most popular 8000 words from the dataset
# Encoding the words in sentences with their key from Tokenizers and padding the encodings with small lengths than 140
# Max character length in sentence is set to 140
top_words = 8000
tokenizer = Tokenizer(num_words = top_words, oov_token="#OOV")
tokenizer.fit_on_texts(x_train)
list_tokenized_train = tokenizer.texts_to_sequences(x_train)

max_review_length = 140
X_train = pad_sequences(list_tokenized_train, maxlen = max_review_length)
Y_train = y_train


# In[36]:


# Encoding the text to sequences using tokenizer to prepare data for neural network
test_word_list = tokenizer.texts_to_sequences(x_test)
X_test = pad_sequences(test_word_list, maxlen = max_review_length)
Y_test = y_test


# In[37]:


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=8000, ngram_range=(2,2))
x_stem = cv.fit_transform(X).toarray()
x_stem


# # Logistic Regression

# In[38]:


x_train, x_test, y_train, y_test = train_test_split(x_stem, Y, test_size = 0.2, random_state = 42)


# In[39]:


lr = LogisticRegression(max_iter=5000)
lr.fit(x_train, y_train)

y_pred_train = lr.predict(x_train)
y_pred_test = lr.predict(x_test)


# In[40]:


print("Train set Accuracy: ",lr.score(x_train, y_train))
print("Test set Accuracy: ", lr.score(x_test, y_test))
print("Train set f1-score: ",f1_score(lr.predict(x_train), y_train))
print("Test set f1-score: ", f1_score(lr.predict(x_test), y_test))
from sklearn.metrics import confusion_matrix
confusion_matrix(y_train, y_pred_train)


# # Naive Bayes

# In[41]:


gnb = GaussianNB()
gnb.fit(x_train, y_train)


# In[42]:


y_pred_train = gnb.predict(x_train)
y_pred_test = gnb.predict(x_test)


# In[43]:


print("Train set Accuracy: ",accuracy_score(y_pred=y_pred_train, y_true=y_train)) 
print("Test set Accuracy: ", accuracy_score(y_pred=y_pred_test, y_true=y_test)) 
print("Train set f1-score: ",f1_score(gnb.predict(x_train), y_train))
print("Test set f1-score: ", f1_score(gnb.predict(x_test), y_test)) 
from sklearn.metrics import confusion_matrix 
confusion_matrix(y_train, y_pred_train)


# # Random Forest Claasifier

# In[44]:


rfc = RandomForestClassifier(n_estimators = 60)
rfc.fit(x_train, y_train)


# In[45]:


print("Train set Accuracy: ",rfc.score(x_train, y_train))
print("Test set Accuracy: ", rfc.score(x_test, y_test))
print("Train set f1-score: ",f1_score(rfc.predict(x_train), y_train))
print("Test set f1-score: ", f1_score(rfc.predict(x_test), y_test))
from sklearn.metrics import confusion_matrix
confusion_matrix(y_train, y_pred_train)


# # Linear SVM

# In[46]:


SVM = LinearSVC()
SVM.fit(x_train, y_train)


# In[47]:


y_pred_train = SVM.predict(x_train)
y_pred_test = SVM.predict(x_test)


# In[48]:


print("Train set Accuracy: ",accuracy_score(y_pred=y_pred_train, y_true=y_train))
print("Test set Accuracy: ", accuracy_score(y_pred=y_pred_test, y_true=y_test))
print("Train set f1-score: ",f1_score(SVM.predict(x_train), y_train))
print("Test set f1-score: ", f1_score(SVM.predict(x_test), y_test))
from sklearn.metrics import confusion_matrix
confusion_matrix(y_train, y_pred_train)


# # k - Neighbour

# In[61]:


KNN = KNeighborsClassifier(n_neighbors = 5)
KNN.fit(x_train, y_train)


# In[62]:


y_pred_train = KNN.predict(x_train)
y_pred_test = KNN.predict(x_test)


# In[63]:


print("Train set Accuracy: ",accuracy_score(y_pred=y_pred_train, y_true=y_train))
print("Test set Accuracy: ", accuracy_score(y_pred=y_pred_test, y_true=y_test))
print("Train set f1-score: ",f1_score(KNN.predict(x_train), y_train))
print("Test set f1-score: ", f1_score(KNN.predict(x_test), y_test))
from sklearn.metrics import confusion_matrix
confusion_matrix(y_train, y_pred_train)


# In[68]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple
n_groups = 4
score_MNB = (92.06,	96.63,	97.82,	85.28)
score_LR = (97.97,	99.22,	96.02,	99.11)
score_LSVC=(99.56,	99.83,	99.12,	98.02)
score_RFC=(92.06,	96.63,	92.00,	85.28) 


#n1=(score_MNB[0], score_LR[0], score_LSVC[0], score_RF[0]) 
#n2=(score_MNB[1], score_LR[1], score_LSVC[1], score_RF[1]) 
#n3=(score_MNB[2], score_LR[2], score_LSVC[2], score_RF[2])
#n4=(score_MNB[3], score_LR[3], score_LSVC[3], score_RF[3])
#n5=(score_MNB[4], score_LR[4], score_LSVC[4], score_RF[4])
fig, ax = plt.subplots()
index = np.arange(n_groups) 
bar_width = 0.1
opacity = 0.7
error_config = {'ecolor': '0.3'}
rects1 = ax.bar(index,score_MNB, bar_width, alpha=opacity, color='b',error_kw=error_config, label='Multinomial Naive Bayes')
z=index + bar_width
rects2 = ax.bar(z, score_LR, bar_width, alpha=opacity, color='r', error_kw=error_config, label='Logistic Regression')
z=z+ bar_width
rects3 = ax.bar(z, score_LSVC, bar_width, alpha=opacity, color='y', error_kw=error_config, label='Linear SVM')
z=z+ bar_width
rects4 = ax.bar(z, score_RFC, bar_width, alpha=opacity, color='g', error_kw=error_config, label='Random Forest')
ax.set_xlabel('Score Parameters')
ax.set_ylabel('Scores (in %)') 
ax.set_title('Scores of Classifiers') 
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('F1', 'Accuracy', 'Precision', 'Recall')) 
ax.legend(bbox_to_anchor=(1,1.02), loc=4, borderaxespad=0) 
fig.tight_layout()
plt.show()


# In[53]:


from textblob import TextBlob
df_new.columns


# In[75]:


df_new['reviews']= df_new['reviews'].astype(str) # Make sure about the correct data type
pol = lambda x: TextBlob(x).sentiment.polarity
df_new['polarity'] = df_new['reviews'].apply(pol) 
num_bins = 50
plt.figure(figsize=(10,6))
n, bins, patches = plt.hist(df_new.polarity, num_bins, facecolor='green', alpha=0.5)
plt.xlabel('Polarity')
plt.ylabel('Number of Reviews')
plt.title('Histogram of Polarity Score')


# In[ ]:





# In[ ]:




