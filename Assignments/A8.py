#!/usr/bin/env python
# coding: utf-8

# In[13]:


########################################################################################
# Name:         Mustafa Furkan BEKER
# Student ID:   61210007
# Department:   Electrical and Electronics Engineering
# Assignment ID: A2
########################################################################################
########################################################################################
# QUESTION I
########################################################################################
print("\n")
print("SOLUTION OF QUESTION I:")
print("****************************")


# In[1]:


#Q1.a

import pandas as pd

url = "http://www.akyokus.com/COE-101-Grades2.xlsx"
df = pd.read_excel(url)
print(df)


# In[2]:



#Q1.b

import pandas as pd

midterm1_weight = 0.25
midterm2_weight = 0.25
final_weight = 0.5

df['Weighted Average'] = (df['MIDI'].fillna(0) * midterm1_weight +
                          df['MIDII'].fillna(0) * midterm2_weight +
                          df['FIN'] * final_weight)

print(df)


# In[3]:



#Q1.c

import pandas as pd

averages = df.mean()
print(averages)


# In[4]:



#Q1.d

import pandas as pd

x = df.fillna(value=0)
print(x)


# In[5]:



#Q1.e

import pandas as pd

def assign_grade(score):
    if score >= 94:
        return "A+"
    elif score >= 87:
        return "A"
    elif score >= 79:
        return "B+"
    elif score >= 70:
        return "B"
    elif score >= 60:
        return "C+"
    elif score >= 50:
        return "C"
    elif score >= 45:
        return "D+"
    elif score >= 40:
        return "D"
    else:
        return "F"

df['GRADE'] = df['FIN'].apply(assign_grade)
print(df)


# In[6]:



#Q1.f

failed_students = df[df['GRADE'] > 'C+']
print(failed_students)


# In[7]:


#Q1.g

passed_students = df[df['GRADE'] < 'C+']
print(passed_students)


# In[8]:


import pandas as pd

pivot_table = pd.pivot_table(df, values='NAME', index='GRADE', aggfunc='count')
pivot_table = pivot_table.reindex(['A', 'A+', 'B', 'B+', 'C', 'C+', 'D', 'D+', 'F'])
pivot_table.rename(columns={'NAME': 'Count'}, inplace=True)
print(pivot_table)


# In[9]:


#Q1.i

import pandas as pd
import matplotlib.pyplot as plt

grade_counts = df['GRADE'].value_counts().sort_index()
grades = ['A', 'A+', 'B', 'B+', 'C', 'C+', 'D', 'D+', 'F']
counts = [grade_counts.get(grade, 0) for grade in grades]

plt.bar(grades, counts)
plt.xlabel('Grade')
plt.ylabel('Count')
plt.title('Count of Students per Grade Category')
plt.show()


# In[10]:


#Q1.j

import pandas as pd
import matplotlib.pyplot as plt

grade_counts = df['GRADE'].value_counts().sort_index()
grades = grade_counts.index.tolist()
counts = grade_counts.values.tolist()

plt.pie(counts, labels=grades, autopct='%1.1f%%')
plt.title('Count of Students per Grade Category')
plt.show()


# In[11]:



#Q1.k

import pandas as pd

correlation_midi_midii = df['MIDI'].corr(df['MIDII'])
correlation_midii_final = df['MIDII'].corr(df['FIN'])

print("Correlation between MID I and MID II:", correlation_midi_midii)
print("Correlation between MID II and FIN:", correlation_midii_final)


# In[12]:


import pandas as pd

midi_grades = df['MIDI']
midii_grades = df['MIDII']
final_grades = df['FIN']

average_grades = (midi_grades.mean() + midii_grades.mean()) / 2
grade_increase_threshold = average_grades * 1.1

students_with_grade_increase = df[df['FIN'] > grade_increase_threshold]

if students_with_grade_increase.empty:
    print("No students found with a grade increase of more than 10%.")
else:
    print("Students with a grade increase of more than 10%:")
    print(students_with_grade_increase[['NAME', 'STUDENTNO']])


# In[14]:


########################################################################################
# Name:         Mustafa Furkan BEKER
# Student ID:   61210007
# Department:   Electrical and Electronics Engineering
# Assignment ID: A2
########################################################################################
########################################################################################
# QUESTION II
########################################################################################
print("\n")
print("SOLUTION OF QUESTION II:")
print("****************************")


# In[15]:



import string
import requests
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize


url = 'https://www.gutenberg.org/files/1342/1342-0.txt'
response = requests.get(url)
text = response.text

start_index = text.index('*** START OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***')
end_index = text.index('*** END OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***')
text = text[start_index:end_index]
sentences = sent_tokenize(text)
words = [word.lower() for word in word_tokenize(text) if word not in string.punctuation]


word_count = len(words)
character_count = sum(len(word) for word in words)
average_word_length = character_count / word_count
average_sentence_length = word_count / len(sentences)
word_distribution = Counter(words)
longest_words = sorted(word_distribution.keys(), key=len, reverse=True)[:10]


print(f"Total word count: {word_count}")
print(f"Total character count: {character_count}")
print(f"Average word length: {average_word_length:.2f}")
print(f"Average sentence length: {average_sentence_length:.2f}")
print("Word distribution:")
for word, count in word_distribution.items():
    print(f"{word}: {count}")
print("Top 10 longest words:")
for word in longest_words:
    print(word)


# In[16]:


import string
import requests
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize


url = 'https://www.gutenberg.org/files/1342/1342-0.txt'
response = requests.get(url)
text = response.text

start_index = text.index('*** START OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***')
end_index = text.index('*** END OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***')
text = text[start_index:end_index]


words = [word.lower() for word in word_tokenize(text) if word not in string.punctuation]


word_frequencies = Counter(words)


top_200_words = dict(word_frequencies.most_common(200))

wordcloud = WordCloud(colormap='prism', background_color='white')

wordcloud = wordcloud.fit_words(top_200_words)


wordcloud.to_file('PrideAndPrejudice.png')


plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[17]:


########################################################################################
# Name:         Mustafa Furkan BEKER
# Student ID:   61210007
# Department:   Electrical and Electronics Engineering
# Assignment ID: A2
########################################################################################
########################################################################################
# QUESTION III
########################################################################################
print("\n")
print("SOLUTION OF QUESTION III:")
print("****************************")


# In[18]:



#Q3.a
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from wordcloud import WordCloud


response = requests.get('https://www.python.org')
soup = BeautifulSoup(response.content, 'html5lib')
text = soup.get_text(strip=True)

wordcloud = WordCloud().generate(text)


plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[19]:



#Q3.b
import requests
from bs4 import BeautifulSoup
import spacy


response = requests.get('https://www.python.org')
soup = BeautifulSoup(response.content, 'html5lib')
text = soup.get_text(strip=True)

nlp = spacy.load('en_core_web_sm')
doc = nlp(text)

sentences = list(doc.sents)
words = [token.text for token in doc]
noun_phrases = [chunk.text for chunk in doc.noun_chunks]

print("Sentences:")
for sentence in sentences:
    print(sentence)

print("\nWords:")
for word in words:
    print(word)

print("\nNoun Phrases:")
for noun_phrase in noun_phrases:
    print(noun_phrase)


# In[20]:



#Q3.c
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob


response = requests.get('https://jamesclear.com/creative-thinking')
soup = BeautifulSoup(response.content, 'html.parser')
text = soup.get_text(strip=True)
blob = TextBlob(text)


print("Sentiment for the entire TextBlob:")
print("Polarity:", blob.sentiment.polarity)
print("Subjectivity:", blob.sentiment.subjectivity)

print("\nSentiment for each Sentence:")
for sentence in blob.sentences:
    print(sentence)
    print("Polarity:", sentence.sentiment.polarity)
    print("Subjectivity:", sentence.sentiment.subjectivity)
    print()


# In[21]:



#Q3.d
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer


response = requests.get('https://jamesclear.com/creative-thinking')
soup = BeautifulSoup(response.content, 'html.parser')
text = soup.get_text(strip=True)
blob = TextBlob(text, analyzer=NaiveBayesAnalyzer())


print("Sentiment for the entire TextBlob:")
print("Classification:", blob.sentiment.classification)
print("Polarity:", blob.sentiment.p_pos - blob.sentiment.p_neg)


print("\nSentiment for each Sentence:")
for sentence in blob.sentences:
    print(sentence)
    print("Classification:", sentence.sentiment.classification)
    print("Polarity:", sentence.sentiment.p_pos - sentence.sentiment.p_neg)
    print()


# In[22]:


#Q3.e
import requests
from bs4 import BeautifulSoup
import spacy


response = requests.get('https://jamesclear.com/creative-thinking')
soup = BeautifulSoup(response.content, 'html.parser')
text = soup.get_text(strip=True)

nlp = spacy.load('en_core_web_sm')
doc = nlp(text)

print("Named Entities:")
for entity in doc.ents:
    print(entity.text, "-", entity.label_)


# In[23]:


#Q3.f
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


article_urls = [
    'https://jamesclear.com/five-step-creative-process',
    'https://jamesclear.com/creative-thinking',
]

articles = []

for url in article_urls:
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text = soup.get_text(strip=True)
    articles.append(text)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(articles)
similarity_matrix = cosine_similarity(tfidf_matrix)

for i in range(len(articles)):
    for j in range(i + 1, len(articles)):
        similarity_score = similarity_matrix[i][j]
        print(f"Similarity between Article {i+1} and Article {j+1}: {similarity_score}")


# In[24]:


########################################################################################
# Name:         Mustafa Furkan BEKER
# Student ID:   61210007
# Department:   Electrical and Electronics Engineering
# Assignment ID: A2
########################################################################################
########################################################################################
# QUESTION IV
########################################################################################
print("\n")
print("SOLUTION OF QUESTION IV:")
print("****************************")


# In[25]:



import pandas as pd
import matplotlib.pyplot as plt

data_url = 'http://www.akyokus.com/ave_hi_la_jan_1895-2018.csv'

def plot_temperature_trends(data_url):
    df = pd.read_csv(data_url)
    df['Year'] = df['Date'] // 100
    df['Month'] = df['Date'] % 100
    df = df[(df['Year'] >= 1895) & (df['Year'] <= 2018)]
    
    la_temperatures = df['Value']
    nyc_temperatures = df['Anomaly']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['Year'], la_temperatures, label='Los Angeles')
    ax.plot(df['Year'], nyc_temperatures, label='New York City')
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Temperature')
    ax.set_title('Temperature Trends in Los Angeles and New York City')
    ax.legend()
    
    plt.show()

plot_temperature_trends(data_url)


# In[ ]:




