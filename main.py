import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

westRootDir = "./res/train/west"
eastRootDir = "./res/train/east"

lyrics = []
coast = []

for subdir, dirs, files in os.walk(westRootDir):
	for f in files:
		pth = os.path.join(subdir, f)
		lyrics.append(open(pth).read())
		coast.append('W')
for subdir, dirs, files in os.walk(eastRootDir):
	for f in files:
		pth = os.path.join(subdir, f)
		lyrics.append(open(pth).read())
		coast.append('E')

countVec = CountVectorizer()
trainCount = countVec.fit_transform(lyrics)

tfidfTransformer = TfidfTransformer(use_idf=False).fit(trainCount)
tfidfTrain = tfidfTransformer.transform(trainCount)

classifier = MultinomialNB().fit(tfidfTrain, coast)

