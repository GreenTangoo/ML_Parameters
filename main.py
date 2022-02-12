from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import urllib.parse

import os
import sys

import KNeighbors
import LogisticRegression
import DecisionTree
import SupportVectorClassification
import MultiLayerNeuronClassifier

from ResultsInterpretation import JsonWriter, PlotsCreator


def load_file(name):
    directory = str(os.getcwd())
    filepath = os.path.join(directory, name)
    with open(filepath,'r') as f:
        data = f.readlines()
    data = list(set(data))
    result = []
    for d in data:
        d = str(urllib.parse.unquote(d))   #converting url encoded data to simple string
        result.append(d)
    return result

def crop_queries(badQueries, validQueries, badThreshold, validThreshold):
    badQueries = badQueries[:badThreshold]
    validQueries = validQueries[:validThreshold]

    return badQueries, validQueries

class Emulator:
    def __init__(self):
        self.methodModelingDict = {
            'knn' : self.__modelingKNeighborsMethod,
            'lgs' : self.__modelingLogisticRegressionMethod,
            'dt' : self.__modelingDecisionTreeMethod,
            'svc' : self.__modelingSupportVectorClassificationMethod,
            'mlpc' : self.__modelingMultiLayerPerceptronClassificationMethod
        }

    def prepareMlSet(self, badThreshold, validThreshold):
        badQueries = load_file('badqueries.txt')
        validQueries = load_file('goodqueries.txt')

        badQueries = list(set(badQueries))
        validQueries = list(set(validQueries))

        badQueries, validQueries = crop_queries(badQueries, validQueries, badThreshold, validThreshold)

        allQueries = badQueries + validQueries
        yBad = [1 for i in range(0, len(badQueries))]  #labels, 1 for malicious and 0 for clean
        yGood = [0 for i in range(0, len(validQueries))]
        y = yBad + yGood

        vectorizer = TfidfVectorizer(min_df = 0.0, analyzer="char", sublinear_tf=True, ngram_range=(1,3)) #converting data to vectors
        X = vectorizer.fit_transform(allQueries)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #splitting data

        self.Xtrain = X_train
        self.Xtest = X_test
        self.yTrain = y_train
        self.yTest = y_test

        self.badQueries = badQueries
        self.validQueries = validQueries

    def modelingWithMethod(self, method):
        return self.methodModelingDict[method]()

    def __modelingKNeighborsMethod(self):
        kNeighborsModel = KNeighbors.KneighboorClassifier(self.Xtrain, self.Xtest, self.yTrain, self.yTest)

        kNeighborsModel.startModeling()

        kNeighborsResultJson = { "KNN" : kNeighborsModel.getJsonRepresentation() }
        return kNeighborsResultJson


    def __modelingLogisticRegressionMethod(self):
        logisticRegressionModel = LogisticRegression.LogistiRegressionClassifier(self.Xtrain, self.Xtest, self.yTrain, self.yTest)

        logisticRegressionModel.setBadQueriesAmount(len(self.badQueries))
        logisticRegressionModel.setValidQueriesAmount(len(self.validQueries))

        logisticRegressionModel.startModeling()

        lgsResultJson = { "LGS" : logisticRegressionModel.getJsonRepresentation() }
        return lgsResultJson

    def __modelingDecisionTreeMethod(self):
        decisionTreeModel = DecisionTree.DTreeClassifier(self.Xtrain, self.Xtest, self.yTrain, self.yTest)

        decisionTreeModel.startModeling()

        dtResultJson = { "DT" : decisionTreeModel.getJsonRepresentation() }
        return dtResultJson

    def __modelingSupportVectorClassificationMethod(self):
        svcModel = SupportVectorClassification.SVCClassifier(self.Xtrain, self.Xtest, self.yTrain, self.yTest)

        svcModel.startModeling()

        svcResultJson = { "SVC" : svcModel.getJsonRepresentation() }
        return svcResultJson

    def __modelingMultiLayerPerceptronClassificationMethod(self):
        mlpModel = MultiLayerNeuronClassifier.MultiLayerPerceptronClassifier(self.Xtrain, self.Xtest, self.yTrain, self.yTest)

        mlpModel.startModeling()

        mlpResultJson = { "MLPC" : mlpModel.getJsonRepresentation() }
        return mlpResultJson


def main():
    if len(sys.argv) != 2:
        print("Incorrect script params. Usage: <script_name>.py <machine_learning_method>")
        print("Methods: knn, lgs, dt, svc, mlpc")
        return

    methodType = sys.argv[1]

    thresholds = [(100, 1000)]
    emulator = Emulator()

    for oneThresholdPair in thresholds:
        badThreshold = oneThresholdPair[0]
        validThreshold = oneThresholdPair[1]

        emulator.prepareMlSet(badThreshold, validThreshold)

        jsonContainer = emulator.modelingWithMethod(methodType)

        serializer = JsonWriter()

        serializer.write(jsonContainer, badThreshold)

        plotCreator = PlotsCreator()
        plotCreator.createPlot(jsonContainer, algorithmType=methodType)



if __name__ == '__main__':
    main()
