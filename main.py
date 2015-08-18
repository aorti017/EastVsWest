import os, string, codecs
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
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


countVec = CountVectorizer()
trainCount = countVec.fit_transform(lyrics)

tfidfTransformer = TfidfTransformer().fit(trainCount)
tfidfTrain = tfidfTransformer.transform(trainCount)

classifier = MultinomialNB().fit(tfidfTrain, coast)

for subdir, dirs, files in os.walk(westTestDir):
	for f in files:
		print f
		pth = os.path.join(subdir, f)
		testSet.append(open(pth).read())
for subdir, dirs, files in os.walk(eastTestDir):
	for f in files:
		print f
		pth = os.path.join(subdir, f)
		testSet.append(open(pth).read())

testCount = countVec.transform(testSet)
tfidfTest = tfidfTransformer.transform(testCount)

predicted = classifier.predict(tfidfTest)
print predicted
