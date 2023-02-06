"""Import library"""
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from datetime import datetime

start_time = datetime.now()

"""Load data"""
input_path = 'C:\My Files\\3\ML\Fake news Detection'
fake = pd.read_csv(os.path.join(input_path, 'Fake.csv'))
real = pd.read_csv(os.path.join(input_path, 'True.csv'))
pd.set_option('display.max_columns', None)

"""Data cleaning"""
fake['label'] = 'fake'
real['label'] = 'real'
data = pd.concat([fake, real], axis=0)
# data = data.drop('subject', axis=1)

"""Split data"""
x_train, x_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.25)
# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)

"""Tfidf Vectorizer"""
tfidf = TfidfVectorizer(stop_words='english')
x_train = tfidf.fit_transform(x_train)
x_test = tfidf.transform(x_test)
# print(x_test)
# print(x_train)

"""Fit model"""
model = RandomForestClassifier()
model.fit(x_train, y_train)

"""Predict model"""
y_pred = model.predict(x_test)
print(y_pred)

"""Calculating accuracy score"""
accuracy = accuracy_score(y_test, y_pred)
print("accuracy: {:.2f}".format(accuracy * 100))

"""Confiusion matrix"""
cm = confusion_matrix(y_test, y_pred)
end_time = datetime.now()

print('Confusion Matrix for RandomForestClassifier')
plot_confusion_matrix(conf_mat=cm, show_absolute=True,
                      show_normed=True,
                      colorbar=True, class_names=['FAKE', 'REAL'], figsize=(10, 10))
plt.title('Random Forest Classifier\n')
plt.show()
print('Duration: {}'.format(end_time - start_time))
