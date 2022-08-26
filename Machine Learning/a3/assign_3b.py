"""
Testing Naive Bayes classification against 8 different representation of text.
word vs stem. Multinomial vs Bernoulli NB. word removal vs no word removal. With 3 different parameters: stemming, word removal, and N-gram.

Jess Nguyen
"""

import numpy as np
import xml.etree.ElementTree as et
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
import gc

gc.enable()


def read_reuters(path="Reuters21578", limit=21578):
    ''' Reads the Reuters-21578 corpus from the given path. (This is assumed to
    be the cleaned-up version of the corpus provided for this code.) The limit
    parameter can be used to stop reading after a certain number of documents
    have been read. - Ref:Sam Scott'''

    def get_dtags(it, index):
        '''Helper function to parse the <D> elements - Code Reference:Sam Scott'''
        dtags = []
        while it[index+1].tag == "D":
            dtags.append(it[index+1].text)
            index += 1
        return dtags, index

    docs = []
    numdocs = 0

    for i in range(22):
        pad = ""
        if i < 10:
            pad = "0"
        print("Reading", path+'\\reut2-0'+pad+str(i)+'.sgm')

        tree = et.parse(path+'\\reut2-0'+pad+str(i)+'.sgm')
        root = tree.getroot()

        it = list(tree.iter())

        index = 0
        while index < len(it):
            if it[index].tag == "REUTERS":
                if numdocs == limit:
                    return docs
                docs.append({})
                numdocs += 1
            elif it[index].tag.lower() in ["topics", "places", "people", "orgs", "exchanges", "companies"]:
                docs[numdocs-1][it[index].tag.lower()], index = get_dtags(it, index)
            elif numdocs > 0:
                docs[numdocs-1][it[index].tag.lower()] = it[index].text

            index += 1

    return docs


def get_labels(docs, labeltype):
    ''' Returns a sorted list of labels from a list of documents that have
    been parsed using read_reuters. The labeltype parameter can be "topics",
    "people", "places", "exchanges", or "orgs". - Code Reference:Sam Scott'''
    import operator

    labels = {}
    try:
        for doc in docs:
            for label in doc[labeltype]:
                labels[label] = 1 + labels.get(label, 0)
    except:
        print("WARNING: '"+labeltype+"' not found.")

    return sorted(labels.items(), key=operator.itemgetter(1), reverse=True)


def createVocabList(dataSet):
    """ dataSet is a list of word lists. Returns the set of words in the dataSet
    as a list. - Code Reference:Sam Scott"""
    vocabSet = set()  # create empty set
    for document in dataSet:
        vocabSet = vocabSet.union(set(document))  # union of the two sets
    return list(vocabSet)


def bagOfWords(vocabList, inputList):
    """ vocabList is a set of words (as a list). inputList is a list of words
    occurring in a document. Returns a list of integers indicating how many
    times each word in the vocabList occurs in the inputList - Code Reference:Sam Scott"""
    d = {}
    for word in inputList:
        d[word] = d.get(word, 0)+1
    bagofwords = []
    for word in vocabList:
        bagofwords.append(d.get(word, 0))
    return bagofwords


def textParse(text):
    """ A utility to split a string into a list of lowercase words, removing punctuation - Code Reference:Sam Scott"""
    import string
    test = ""
    for word in text.split():
        test = test + str(word.lower().strip(string.punctuation))

    return [word.lower().strip(string.punctuation) for word in text.split()]


def setOfWords(vocabList, inputList):
    """ vocabList is a set of words (as a list). inputList is a list of words
    occurring in a document. Returns a list of 1's and 0's to indicate
    the presence or absence of each word in vocabList - Code Reference:Sam Scott"""
    d = {}
    for word in inputList:
        d[word] = 1
    setofwords = []
    for word in vocabList:
        setofwords.append(d.get(word, 0))
    return setofwords


def createDataSet(dataArray, wordsType, vocabType, fillArray, dataType):
    """
    To set the data array for different parameters of confusion matrics
    """
    totalIndex = len(dataArray)
    if wordsType == "bag":
        for doc in dataArray:
            fillArray.append(bagOfWords(vocabType, doc))
            totalIndex -= 1
            if(totalIndex % 100 == 0):
                print(dataType)
            print(totalIndex)
    elif wordsType == "set":
        for doc in dataArray:
            fillArray.append(setOfWords(vocabType, doc))
            if(totalIndex % 100 == 0):
                print(dataType)
            totalIndex -= 1
            print(totalIndex)


# Extract data and labels
docs = read_reuters()

labels = []
data = []
isProcessingDoc = True
fiveTopics = ["earn", "acq", "bop", "veg-oil", "ship"]
for topic in fiveTopics:
    tempTopicLabels = []
    for doc in range(0, len(docs)):
        if isProcessingDoc:
            body = []
            dateline = []
            title = []
            text = []
            if "body" in docs[doc]:
                body = textParse(docs[doc]['body'])
            if "title" in docs[doc]:
                title = textParse(docs[doc]['title'])
            if "dateline" in docs[doc]:
                dateline = textParse(docs[doc]['dateline'])
            if "text" in docs[doc]:
                text = textParse(docs[doc]['text'])
            compressedDoc = np.array(title + dateline + body + text)
            data.append(compressedDoc)

        if topic in docs[doc]["topics"]:
            tempTopicLabels.append(1)
        else:
            tempTopicLabels.append(0)
    isProcessingDoc = False
    labels.append(tempTopicLabels)

# Split Data
totalRows = len(data)
trainingNumRows = round(totalRows * 0.75)


def classify_reuters(label, dataWord, nb):
    """
        Classify with Multinomial Naive Bayes and Bernoulli.
    """
    # Get the first 75% of the dataWord to be training
    trainingData = np.array(dataWord[:trainingNumRows])
    trainingLabels = np.array(label[:trainingNumRows])

    # Get the last 25% of the dataWord to be testing
    testingData = np.array(dataWord[trainingNumRows:])
    testingLabels = np.array(label[trainingNumRows:])

    nb.fit(trainingData, trainingLabels)
    prediction = nb.predict(testingData)

    confusionMatrix = confusion_matrix(testingLabels, prediction)
    return confusionMatrix


def printMatrix(matrix, ver):
    """
        Display combined matrix of all topics.
    """
    print("\nCOMBINED MATRIX\n" + ver + ":\n", matrix)
    print("\nAccuracy:", round((matrix[1][1] + matrix[0][0]) / (
        matrix[1][1] + matrix[1][0] + matrix[0][0] + matrix[0][1]), 2))
    print("Precision:", round(
        (matrix[1][1] / (matrix[1][1] + matrix[0][1])), 2))
    print("Recall:", round((matrix[1][1] / (matrix[1][1] + matrix[1][0])), 2))


# Initialize
s = PorterStemmer()
bags = []
setsArray = []
stemmedArray = []
stemmedBags = []
vocabularyStopped = []
vocabularyNgram = []
setStemmedBags = []
bagStopped = []
setsStoppedArray = []
bagNgram = []
setsNgramArray = []

combinedMatrixV1 = []
combinedMatrixV2 = []
combinedMatrixV3 = []
combinedMatrixV4 = []
combinedMatrixV5 = []
combinedMatrixV6 = []
combinedMatrixV7 = []
combinedMatrixV8 = []

mnb = MultinomialNB()
bnb = BernoulliNB()

# Create vocabulary
vocab = createVocabList(data)
print("Vocabulary size:", len(vocab))

# LOOP - Bags of word
print("---> Bag of wording the data...")
createDataSet(data, "bag", vocab, bags, "---> Bag of wording the data...")
gc.collect()
index = 0
for topicLabels in labels:
    print("Computing...", fiveTopics[index])
    # Bags
    combinedMatrixV1.append(classify_reuters(topicLabels, bags, mnb))
    print("....")
    index += 1
del bags


# LOOP - Set of words
print("---> Set of wording the data...")
createDataSet(data, "set", vocab, setsArray, "---> Set of wording the data...")
gc.collect()
index = 0
for topicLabels in labels:
    print("Computing...", fiveTopics[index])
    # Set bags
    combinedMatrixV2.append(classify_reuters(topicLabels, setsArray, bnb))
    print(".....")
    index += 1
del setsArray

# LOOP - NLTK for stemming
print("---> Stemming data...")
for row in data:
    tempArray = []
    for el in row:
        tempArray.append(s.stem(el))
    stemmedArray.append(tempArray)
vocabStemmed = createVocabList(stemmedArray)
gc.collect()

# LOOP - Stemmed before bagging words
print("---> Stemming then bag of words the data...")
createDataSet(stemmedArray, "bag", vocabStemmed, stemmedBags,
              "---> Stemming then bag of words the data...")
gc.collect()
index = 0
for topicLabels in labels:
    print("Computing...", fiveTopics[index])
    # Stemmed bags
    combinedMatrixV3.append(classify_reuters(topicLabels, stemmedBags, mnb))
    print("......")
    index += 1
del stemmedBags

# VEC LOOP - Stopped words
vectorizer = CountVectorizer(
    token_pattern=r"(?u)\b\w+\b", stop_words="english")
print("---> Removing stopped words from data...")
for row in data:
    vec = vectorizer.fit_transform(row)
    vocabularyStopped.append(sorted(vectorizer.vocabulary_))
vocabStop = createVocabList(vocabularyStopped)
gc.collect()

# VEC LOOP - 2 gram
vectorizer = CountVectorizer(
    token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2), stop_words=None)
print("---> 2 gramming the words...")
for row in data:
    vec = vectorizer.fit_transform(row)
    vocabularyNgram.append(sorted(vectorizer.vocabulary_))
vocabNgram = createVocabList(vocabularyNgram)
gc.collect()
del data

# LOOP - Stemmed set of words
print("---> Stemmed the set of words from data...")
createDataSet(stemmedArray, "set", vocabStemmed, setStemmedBags,
              "---> Stemmed the set of words from data...")
gc.collect()
index = 0
for topicLabels in labels:
    print("Computing...", fiveTopics[index])
    # Set stemmed bags
    combinedMatrixV4.append(classify_reuters(topicLabels, setStemmedBags, bnb))
    print(".......")
    index += 1
del setStemmedBags
del stemmedArray

# LOOP - Stopped bag of words
print("---> Removing stopped words the bag of words from data...")
createDataSet(vocabularyStopped, "bag", vocabStop, bagStopped,
              "---> Removing stopped words the bag of words from data...")
gc.collect()
index = 0
for topicLabels in labels:
    print("Computing...", fiveTopics[index])
    # Bag stopped words
    combinedMatrixV5.append(classify_reuters(topicLabels, bagStopped, mnb))
    print("........")
    index += 1
del bagStopped

# LOOP - Stopped set of words
print("---> Removing stopped words the set of words from data...")
createDataSet(vocabularyStopped, "set", vocabStop, setsStoppedArray,
              "---> Removing stopped words the set of words from data...")
gc.collect()
index = 0
for topicLabels in labels:
    print("Computing...", fiveTopics[index])
    # Set bag stopped words
    combinedMatrixV6.append(classify_reuters(
        topicLabels, setsStoppedArray, bnb))
    print(".........")
    index += 1
del setsStoppedArray
del vocabStop
del vocabularyStopped

# LOOP - 2 gram bag of words
print("---> 2 gramming the bag of words...")
createDataSet(vocabularyNgram, "bag", vocabNgram, bagNgram,
              "---> 2 gramming the bag of words...")
gc.collect()
index = 0
for topicLabels in labels:
    print("Computing...", fiveTopics[index])
    # N-gram w/ bags
    combinedMatrixV7.append(classify_reuters(topicLabels, bagNgram, mnb))
    print("..........")
    index += 1
del bagNgram

# LOOP - 2 gram set of words
print("---> 2 gramming the set of words...")
createDataSet(vocabularyNgram, "set", vocabNgram, setsNgramArray,
              "---> 2 gramming the set of words...")
gc.collect()
index = 0
for topicLabels in labels:
    print("Computing...", fiveTopics[index])
    # N-grams w/ set
    combinedMatrixV8.append(classify_reuters(topicLabels, setsNgramArray, bnb))
    print("...........")
    index += 1
del setsNgramArray
del topicLabels
del vocabNgram
del vocabularyNgram

# Printing overall results
combinedMatrixV1 = np.array(
    combinedMatrixV1[0]+combinedMatrixV1[1]+combinedMatrixV1[2]+combinedMatrixV1[3]+combinedMatrixV1[4])
printMatrix(combinedMatrixV1, "v1 no word removal w/ bags w/ Multi")
del combinedMatrixV1

combinedMatrixV2 = np.array(
    combinedMatrixV2[0]+combinedMatrixV2[1]+combinedMatrixV2[2]+combinedMatrixV2[3]+combinedMatrixV2[4])
printMatrix(combinedMatrixV2, "v2 no word removal w/ set bags w/ Bernoulli")
del combinedMatrixV2

combinedMatrixV3 = np.array(
    combinedMatrixV3[0]+combinedMatrixV3[1]+combinedMatrixV3[2]+combinedMatrixV3[3]+combinedMatrixV3[4])
printMatrix(combinedMatrixV3, "v3 no word removal w/ stemmed bags w/ Multi")
del combinedMatrixV3

combinedMatrixV4 = np.array(
    combinedMatrixV4[0]+combinedMatrixV4[1]+combinedMatrixV4[2]+combinedMatrixV4[3]+combinedMatrixV4[4])
printMatrix(combinedMatrixV4,
            "v4 no word removal w/ set stemmed bags w/ Bernoulli")
del combinedMatrixV4

combinedMatrixV5 = np.array(
    combinedMatrixV5[0]+combinedMatrixV5[1]+combinedMatrixV5[2]+combinedMatrixV5[3]+combinedMatrixV5[4])
printMatrix(combinedMatrixV5, "v5 word removal(stopped words) w/ bags w/ Multi")
del combinedMatrixV5

combinedMatrixV6 = np.array(
    combinedMatrixV6[0]+combinedMatrixV6[1]+combinedMatrixV6[2]+combinedMatrixV6[3]+combinedMatrixV6[4])
printMatrix(combinedMatrixV6,
            "v6 word removal(stopped words) w/ set bags w/ Bernoulli")
del combinedMatrixV6

combinedMatrixV7 = np.array(
    combinedMatrixV7[0]+combinedMatrixV7[1]+combinedMatrixV7[2]+combinedMatrixV7[3]+combinedMatrixV7[4])
printMatrix(combinedMatrixV7, "v7 2-gram w/ bags w/ Multi")
del combinedMatrixV7

combinedMatrixV8 = np.array(
    combinedMatrixV8[0]+combinedMatrixV8[1]+combinedMatrixV8[2]+combinedMatrixV8[3]+combinedMatrixV8[4])
printMatrix(combinedMatrixV8, "v8 2-gram w/ set bags w/ Bernoulli")
del combinedMatrixV8
