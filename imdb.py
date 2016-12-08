import pandas
from bs4 import BeautifulSoup
import re
import nltk
#only do next line once
#nltk.download() #download everything except panlex_lite
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from gensim.models import word2vec

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
	reviews = data['review'][0:] #data we are concerned with
	no_html_data = get_data(reviews) #list does not have html tags
	letters_only = get_letters(no_html_data) #list of letters only
	clean_reviews = clean_up(letters_only) #list of cleaned reviews
	word_count = count_words(clean_reviews) #keeps the count for the words encountered

'''
Deep Learning Below
'''
def clean_sentence( raw ):
	bs = BeautifulSoup(raw, "html.parser")
	letters_only = re.sub("[^a-zA-Z]"," ",bs.get_text())
	lower_case = letters_only.lower()
	words = lower_case.split()
	return words

def review_to_sentences( review_data, tokenizer):
	#didn’t seem to work without it, thanks StackOverflow
	#review = review.decode("utf-8")
	#strip out whitespace at beginning and end
	sentences_to_return = []
	for row in review_data:
		review = row.strip()
		raw_sentences = tokenizer.tokenize(review)
		sentences_list = []

		for sentence in raw_sentences:
			if len(sentence) > 0: #skip it if the sentence is empty
				cl_sent = clean_sentence(sentence)
				sentences_list += cl_sent
		sentences_to_return.append(sentences_list)
	return sentences_to_return

def make_attribute_vec(words, model, num_attributes):
	# Pre-initialize an empty numpy array (for speed)
	attribute_vec = numpy.zeros((num_attributes,),dtype="float32")
	nwords = 0.0
	# Loop over each word in the review and, if it is in the model’s
	# vocaublary, add its attribute vector to the total
	for word in words:
		if word in model.vocab:
			nwords = nwords + 1.0
			attribute_vec = numpy.add(attribute_vec,model[word])

	# Divide the result by the number of words to get the average
	attribute_vec = numpy.divide(attribute_vec,nwords)
	return attribute_vec

def deep_learning():
	print("--- Tokenizing the data ---")
	tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
	data = pandas.read_csv("imdb_reviews.tsv", delimiter="\t")
	sentences_for_all_reviews = review_to_sentences(data["review"],tokenizer) 

	print("--- Training the data using Word2Vec ---")
	num_attributes = 300 # Word vector dimensionality
	min_word_count = 40 # Minimum word frequency
	num_workers = 4 # Number of threads to run in parallel
	context = 10 # Context window size
	downsampling = 1e-3 # Downsample setting for frequent words
	# Initialize and train the model (this will take some time)

	model = word2vec.Word2Vec(sentences_for_all_reviews, \
			workers=num_workers, size=num_attributes, \
			min_count = min_word_count, \
			window = context, sample = downsampling)
	#saves memory if you’re done training it
	model.init_sims(replace=True)

	print("--- Print results ---")
	def fun_with_model():
		# model.vocab
		# print("chicago" in model.vocab)
		# print("iowa" in model.vocab)
		# print(model.similarity("england","france"))
		print(model.similarity("england","paris"))
		# model.most_similar("king")
		# model.most_similar("awful")
		# model.doesnt_match(["man","woman","child","kitchen"])
		# model.doesnt_match(["france","england","germany","berlin"])
		# model["king"]
		# model["queen"]
		# model["man"]
		# model["woman"]
		# (model["king"] - model["man"] + model["woman"])
		# model.most_similar(positive=["woman", "king"], negative=["man"])
	fun_with_model()

	def fun_with_attributes():
		print("original: ",data["review"][0])
		clean_review = clean_sentence(data["review"][0])
		print("clean: ",clean_review)
		review_vector = make_attribute_vec(clean_review, model, \
				num_attributes)
		print("vector: ",review_vector)
	fun_with_attributes()

#main()
deep_learning()


