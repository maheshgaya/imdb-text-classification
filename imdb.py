import pandas
from bs4 import BeautifulSoup
import re
import nltk
#only do next line once
#nltk.download() #download everythin except panlex_lite
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

#functions 
def get_data(list):
	"""
	Get the clean text from the tsv file without any html.
	"""
	data_list = []
	for row in list:
		rev_soup = BeautifulSoup(row, "html.parser")
		data_list.append(rev_soup.get_text())
	return data_list

def get_letters(list):
	"""
	Contains letters only.
	Trims all the punctuation.
	"""
	letters_only_list = []
	for row in list:
		letters_only_list.append(re.sub("[^a-zA-Z]"," ", row))
	return letters_only_list

def clean_up(list):
	"""
	Cleans up the reviews.
	Contains no html tags or punctuations.
	"""
	clean_data = []
	for row in list:
		lower_case = row.lower()
		words = lower_case.split()
		stop_words = stopwords.words("english")
		for word in words:
			if word in stop_words:
				words.remove(word)
			clean_text = " ".join(words)
		clean_data.append(clean_text)
	return clean_data

def count_words(list):
	"""
	Counts the number of occurrences of 5000 common words.
	Keeps them in a list of lists.
	"""
	#Bag of Words with 5000 most common words
	vectorizer = CountVectorizer(analyzer="word", \
	max_features = 5000)

	word_columns = vectorizer.fit_transform(list)

	#convert to numpy array so we can feed it
	#into learning algorithm
	word_columns = word_columns.toarray()
	#print(vectorizer.get_feature_names())
	#print(word_columns)
	return word_columns

	
def main():
	data = pandas.read_csv("imdb_reviews.tsv", delimiter="\t") #get the data.  Yay!
	reviews = data['review'][0:1000] #data we are concerned of
	no_html_data = get_data(reviews) #list does not have html tags
	letters_only = get_letters(no_html_data) #list of letters only
	clean_reviews = clean_up(letters_only) #list of cleaned reviews
	word_count = count_words(clean_reviews) #keeps the count for the words encountered
	print(word_count)

main()


