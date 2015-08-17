import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

westTrainDir = "./res/train/west"
eastTrainDir = "./res/train/east"
westTestDir = "./res/test/west"
eastTestDir = "./res/test/east"

lyrics = []
coast = []
testSet = []

for subdir, dirs, files in os.walk(westTrainDir):
	for f in files:
		pth = os.path.join(subdir, f)
		lyrics.append(open(pth).read())
		coast.append('W')
for subdir, dirs, files in os.walk(eastTrainDir):
	for f in files:
		pth = os.path.join(subdir, f)
		lyrics.append(open(pth).read())
		coast.append('E')

countVec = CountVectorizer()
trainCount = countVec.fit_transform(lyrics)

tfidfTransformer = TfidfTransformer(use_idf=False).fit(trainCount)
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
