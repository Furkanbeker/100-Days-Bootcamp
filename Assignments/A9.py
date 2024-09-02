#!/usr/bin/env python
# coding: utf-8

# In[1]:

# QUESTION I
#########################################################################################
print("\n")
print("SOLUTION OF QUESTION I:")
print("********************************************************************************")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV


def main():
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    y = raw_df.values[1::2, 2]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    parameters = {'alpha': np.logspace(-3, 3, 7)}

    ridge = Ridge()
    clf = GridSearchCV(ridge, parameters)
    clf.fit(X_train, y_train)
    best_alpha = clf.best_params_['alpha']
    print('Best alpha:', best_alpha)

    ridge = Ridge(alpha=best_alpha)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)

    plt.scatter(y_test, y_pred)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("True value vs Predicted values - Ridge Regression")
    plt.show()

    poly = PolynomialFeatures(2, interaction_only=True)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    clf = GridSearchCV(ridge, parameters)
    clf.fit(X_train_poly, y_train)
    best_alpha_poly = clf.best_params_['alpha']
    print('Best alpha for polynomial features:', best_alpha_poly)

    ridge = Ridge(alpha=best_alpha_poly)
    ridge.fit(X_train_poly, y_train)
    y_pred_poly = ridge.predict(X_test_poly)

    plt.scatter(y_test, y_pred_poly)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("True value vs Predicted values - Polynomial Ridge Regression")
    plt.show()

    X_train_single = X_train[:, 0].reshape(-1, 1)
    X_test_single = X_test[:, 0].reshape(-1, 1)

    clf = GridSearchCV(ridge, parameters)
    clf.fit(X_train_single, y_train)
    best_alpha_single = clf.best_params_['alpha']
    print('Best alpha for single attribute:', best_alpha_single)

    ridge = Ridge(alpha=best_alpha_single)
    ridge.fit(X_train_single, y_train)
    y_pred_single = ridge.predict(X_test_single)

    plt.scatter(y_test, y_pred_single)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("True value vs Predicted values - Single Attribute Ridge Regression")
    plt.show()


if __name__ == '__main__':
    main()


# In[ ]:


#########################################################################################
# QUESTION II
#########################################################################################
print("\n")
print("SOLUTION OF QUESTION II:")
print("********************************************************************************")

import sklearn.neighbors as nb
import sklearn.metrics as metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def main():
    warnings = __import__('warnings')
    stats = __import__('scipy').stats

    warnings.filterwarnings('ignore', category=FutureWarning)
    stats.mode._keepdims = True

    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist.data, mnist.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    K_values = [1, 3, 5, 7, 9]

    for K in K_values:
        knn = nb.KNeighborsClassifier(n_neighbors=K)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        print("Accuracy with K={}: {}%".format(K, accuracy * 100))

if __name__ == '__main__':
    main()


# In[ ]:


#########################################################################################
# QUESTION III
#########################################################################################
print("\n")
print("SOLUTION OF QUESTION III:")
print("********************************************************************************")

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

def preprocess_images(train_images, test_images):
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    return train_images, test_images

def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    return model

def compile_and_train(model, train_images, train_labels, test_images, test_labels):
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
    return history

def evaluate_model(model, test_images, test_labels):
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    return test_acc

def print_test_accuracy(test_acc):
    print('\nTest accuracy:', test_acc)

def main():
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = preprocess_images(train_images, test_images)
    model = create_model()
    history = compile_and_train(model, train_images, train_labels, test_images, test_labels)
    test_acc = evaluate_model(model, test_images, test_labels)
    print_test_accuracy(test_acc)

if __name__ == '__main__':
    main()


# In[ ]:


#########################################################################################
# QUESTION IV
#########################################################################################
print("\n")
print("SOLUTION OF QUESTION IV:")
print("********************************************************************************")

import sqlite3

def main():
    db_path = "C:\\Users\\mfb36\\Desktop\\books.db"
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    authors = fetch_authors(cur)
    print(authors)

    titles = fetch_titles(cur)
    print(titles)

    books = fetch_books_by_author(cur, 'FirstName', 'LastName')
    print(books)

    insert_new_author(cur, 'NewAuthorFirstName', 'NewAuthorLastName')

    authors_after_insert = fetch_all_authors(cur)
    print('Authors after insert:', authors_after_insert)

    conn.commit()
    conn.close()

def fetch_authors(cur):
    cur.execute("SELECT last FROM authors ORDER BY last DESC")
    authors = cur.fetchall()
    return authors

def fetch_titles(cur):
    cur.execute("SELECT title FROM titles ORDER BY title ASC")
    titles = cur.fetchall()
    return titles

def fetch_books_by_author(cur, first_name, last_name):
    cur.execute("""
    SELECT t.title, t.copyright, t.isbn FROM titles AS t
    INNER JOIN author_ISBN AS ai ON t.isbn = ai.isbn
    INNER JOIN authors AS a ON ai.id = a.id
    WHERE a.first = ? AND a.last = ?
    ORDER BY t.title ASC;
    """, (first_name, last_name))
    books = cur.fetchall()
    return books

def insert_new_author(cur, first_name, last_name):
    cur.execute("INSERT INTO authors (first, last) VALUES (?, ?)", (first_name, last_name))

def fetch_all_authors(cur):
    cur.execute("SELECT * FROM authors")
    authors = cur.fetchall()
    return authors

if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




