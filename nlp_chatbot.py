import pandas as pd
import nltk 
import numpy as np
import re
nltk.download("punkt")
nltk.download("wordnet")
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
from nltk.stem import wordnet # to perform lemmitization
from sklearn.feature_extraction.text import CountVectorizer # to perform bow
from sklearn.feature_extraction.text import TfidfVectorizer # to perform tfidf
from nltk import pos_tag # for parts of speech
from sklearn.metrics import pairwise_distances # to perfrom cosine similarity
from nltk import word_tokenize # to create tokens
from nltk.corpus import stopwords # for stop words

df=pd.read_excel('dialog_talk_agent.xlsx')
df.head(20)

df.shape[0] # returns the number of rows in dataset

df.ffill(axis = 0,inplace=True) # fills the null value with the previous value.

"""faq = pd.read_excel("faq_data.xls")
faq = faq.drop(['Unnamed: 0'], axis=1)

faq.shape[0]

faq = faq.replace(r'^\s*$', np.nan, regex=True)

faq.isnull().sum()

faq.dropna(inplace=True)

faq.isnull().sum()

faq.shape[0]

faq = faq.rename(columns={'Question': 'Context', 'Answer': 'Text Response'})

df = pd.concat([df, faq], ignore_index=True)
"""

# word tokenizing
    
s='tell me about your personality'
words=word_tokenize(s)

lemma = wordnet.WordNetLemmatizer() # intializing lemmatizer
lemma.lemmatize('absorbed', pos = 'v')

pos_tag(nltk.word_tokenize(s),tagset = None) # returns the parts of speech of every word

# function that performs text normalization steps

def text_normalization(text):
    text=str(text).lower() # text to lower case
    spl_char_text=re.sub(r'[^ a-z]','',text) # removing special characters
    tokens=nltk.word_tokenize(spl_char_text) # word tokenizing
    lema=wordnet.WordNetLemmatizer() # intializing lemmatization
    tags_list=pos_tag(tokens,tagset=None) # parts of speech
    lema_words=[]   # empty list 
    for token,pos_token in tags_list:
        if pos_token.startswith('V'):  # Verb
            pos_val='v'
        elif pos_token.startswith('J'): # Adjective
            pos_val='a'
        elif pos_token.startswith('R'): # Adverb
            pos_val='r'
        else:
            pos_val='n' # Noun
        lema_token=lema.lemmatize(token,pos_val) # performing lemmatization
        lema_words.append(lema_token) # appending the lemmatized token into a list
    
    return " ".join(lema_words) # returns the lemmatized tokens as a sentence

text_normalization('telling you some stuff about me')

df['lemmatized_text']=df['Context'].apply(text_normalization) # applying the fuction to the dataset to get clean text
df.tail(15)

# all the stop words we have 

stop = stopwords.words('english')

"""# bag of words"""

cv = CountVectorizer() # intializing the count vectorizer
X = cv.fit_transform(df['lemmatized_text']).toarray()

# returns all the unique word from data 

features = cv.get_feature_names()
df_bow = pd.DataFrame(X, columns = features)
df_bow.head()

Question ='Will you help me and tell me about yourself more' # considering an example query

# checking for stop words

Q=[]
a=Question.split()
for i in a:
    if i in stop:
        continue
    else:
        Q.append(i)
    b=" ".join(Q)

Question_lemma = text_normalization(b) # applying the function that we created for text normalizing
Question_bow = cv.transform([Question_lemma]).toarray() # applying bow

text_normalization

Question_bow

"""# similarity"""

# cosine similarity for the above question we considered.

cosine_value = 1- pairwise_distances(df_bow, Question_bow, metric = 'cosine' )
(cosine_value)

df['similarity_bow']=cosine_value # creating a new column

df_simi = pd.DataFrame(df, columns=['Text Response','similarity_bow']) # taking similarity value of responses for the question we took
df_simi

df_simi_sort = df_simi.sort_values(by='similarity_bow', ascending=False) # sorting the values
df_simi_sort.head()

threshold = 0.2 # considering the value of p=smiliarity to be greater than 0.2
df_threshold = df_simi_sort[df_simi_sort['similarity_bow'] > threshold] 
df_threshold

"""- Finally using bow for the question 'Will you help me and tell me about yourself more' , the above are the responses we got using bow and the smiliarity value of responses, we consider the response with highest similarity"""

index_value = cosine_value.argmax() # returns the index number of highest value
index_value

(Question)

df['Text Response'].loc[index_value] # The text at the above index becomes the response for the question

"""# tf-idf"""

# using tf-idf

tfidf=TfidfVectorizer() # intializing tf-id 
x_tfidf=tfidf.fit_transform(df['lemmatized_text']).toarray() # transforming the data into array

Question1 ='Tell me about yourself.'

Question_lemma1 = text_normalization(Question1)
Question_tfidf = tfidf.transform([Question_lemma1]).toarray() # applying tf-idf

# returns all the unique word from data with a score of that word

df_tfidf=pd.DataFrame(x_tfidf,columns=tfidf.get_feature_names()) 
df_tfidf.head()

"""# similarity"""

cos=1-pairwise_distances(df_tfidf,Question_tfidf,metric='cosine')  # applying cosine similarity
cos

df['similarity_tfidf']=cos # creating a new column 
df_simi_tfidf = pd.DataFrame(df, columns=['Text Response','similarity_tfidf']) # taking similarity value of responses for the question we took
df_simi_tfidf

df_simi_tfidf_sort = df_simi_tfidf.sort_values(by='similarity_tfidf', ascending=False) # sorting the values
df_simi_tfidf_sort.head(10)

threshold = 0.2 # considering the value of p=smiliarity to be greater than 0.2
df_threshold = df_simi_tfidf_sort[df_simi_tfidf_sort['similarity_tfidf'] > threshold] 
df_threshold

"""- by using tfidf for the question 'Will you help me and tell me about yourself more' , the above are the responses we got and the smiliarity value of responses, we consider the response with highest similarity"""

index_value1 = cos.argmax() # returns the index number of highest value
index_value1

Question1

df['Text Response'].loc[index_value1]  # returns the text at that index

"""# Model Using Bag of Words"""

# Function that removes stop words and process the text

def stopword_(text):   
    tag_list=pos_tag(nltk.word_tokenize(text),tagset=None)
    stop=stopwords.words('english')
    lema=wordnet.WordNetLemmatizer()
    lema_word=[]
    for token,pos_token in tag_list:
        if token in stop:
            continue
        if pos_token.startswith('V'):
            pos_val='v'
        elif pos_token.startswith('J'):
            pos_val='a'
        elif pos_token.startswith('R'):
            pos_val='r'
        else:
            pos_val='n'
        lema_token=lema.lemmatize(token,pos_val)
        lema_word.append(lema_token)
    return " ".join(lema_word)

# defining a function that returns response to query using bow

def chat_bow(text):
    s=stopword_(text)
    lemma=text_normalization(s) # calling the function to perform text normalization
    bow=cv.transform([lemma]).toarray() # applying bow
    cosine_value = 1- pairwise_distances(df_bow,bow, metric = 'cosine' )
    index_value=cosine_value.argmax() # getting index value 
    return df['Text Response'].loc[index_value]

"""# Model Using tf-idf"""

# defining a function that returns response to query using tf-idf

def chat_tfidf(text):
    lemma=text_normalization(text) # calling the function to perform text normalization
    tf=tfidf.transform([lemma]).toarray() # applying tf-idf
    cos=1-pairwise_distances(df_tfidf,tf,metric='cosine') # applying cosine similarity
    index_value=cos.argmax() # getting index value 
    return df['Text Response'].loc[index_value]

"""# Conclusion
- Our chat bot worked well with both bow and tf-idf, tf-idf model worked well even with stop words compared to bow where we had to remove stop words before using bag of words.
"""