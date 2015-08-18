import os, string, codecs
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

westTrainDir = "./res/train/west"
eastTrainDir = "./res/train/east"
westTestDir = "./res/test/west"
eastTestDir = "./res/test/east"

lyrics = []
coast = []
testSet = []

stemmer = SnowballStemmer("english")

for subdir, dirs, files in os.walk(westTrainDir):
	for f in files:
		temp = ""
		pth = os.path.join(subdir, f)
		fileLyrics = open(pth).read().decode('utf-8')
		fileLyrics = fileLyrics.replace(string.punctuation, "")
		for i in string.punctuation:
			fileLyrics = fileLyrics.replace(i, '')
		for i in fileLyrics.split(" "):
			temp += stemmer.stem(i) + " "
		lyrics.append(temp)
		coast.append('W')
for subdir, dirs, files in os.walk(eastTrainDir):
	for f in files:
		temp = ""
		pth = os.path.join(subdir, f)
		fileLyrics = open(pth).read().decode('utf-8')
		fileLyrics = fileLyrics.replace(string.punctuation, "")
		for i in fileLyrics.split(" "):
			temp += stemmer.stem(i) + " "
		lyrics.append(temp)
		coast.append('E')

sw = stopwords.words("english")

countVec = CountVectorizer(stop_words=sw)
trainCount = countVec.fit_transform(lyrics)

classifier = MultinomialNB(alpha=.5).fit(trainCount, coast)

for subdir, dirs, files in os.walk(westTestDir):
	for f in files:
		print f
		temp = ""
		pth = os.path.join(subdir, f)
		fileLyrics = open(pth).read().decode('utf-8')
		fileLyrics = fileLyrics.replace(string.punctuation, "")
		for i in fileLyrics.split(" "):
			temp += stemmer.stem(i) + " "
		testSet.append(temp)
for subdir, dirs, files in os.walk(eastTestDir):
	for f in files:
		print f
		temp = ""
		pth = os.path.join(subdir, f)
		fileLyrics = open(pth).read().decode('utf-8')
		fileLyrics = fileLyrics.replace(string.punctuation, "")
		for i in fileLyrics.split(" "):
			temp += stemmer.stem(i) + " "
		testSet.append(temp)

testCount = countVec.transform(testSet)
predicted = classifier.predict(testCount)
print "Accuracy: " + str(accuracy_score(predicted, ['W', 'W ', 'W', 'W ',  'W ',  'W ',  'W ',  'W ', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E']))
