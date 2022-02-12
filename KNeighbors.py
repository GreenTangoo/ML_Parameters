from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics



class KneighboorClassifier:
    def __init__(self, Xtrain, Xtest, yTrain, yTest):
        self.Xtrain = Xtrain
        self.Xtest = Xtest
        self.yTrain = yTrain
        self.yTest = yTest

        self.neighborsStatistic = {}

    def startModeling(self):
        print('Start modeling with kNeighbors model')

        self.testNeighborsParameter()

    def getJsonRepresentation(self):
        return self.neighborsStatistic

    def testNeighborsParameter(self):
        neighborsRange = range(1, 11)

        for neighbors in neighborsRange:
            self.testAlgorithmNearestType(neighbors)


    def testAlgorithmNearestType(self, neighbors):
        algTypes = ['brute']

        for type in algTypes:
            print(f'Modeliing with {neighbors} neighbors | {type} algorithm type')

            knn = KNeighborsClassifier(n_neighbors=neighbors, algorithm=type)
            knn.fit(self.Xtrain, self.yTrain)

            testAccuracy = knn.score(self.Xtest, self.yTest)

            self.neighborsStatistic.update({ neighbors : { type : {'test' : testAccuracy} } })
