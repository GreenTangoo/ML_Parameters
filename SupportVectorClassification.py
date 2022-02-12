from sklearn.svm import SVC
from sklearn import metrics



class SVCClassifier:
    def __init__(self, Xtrain, Xtest, yTrain, yTest):
        self.Xtrain = Xtrain
        self.Xtest = Xtest
        self.yTrain = yTrain
        self.yTest = yTest

        self.svcStatistic = {}

    def startModeling(self):
        print('Start modeling with SupportVectorClassification model')

        self.testKernelType()

    def getJsonRepresentation(self):
        return self.svcStatistic

    def testKernelType(self):
        kernels = ['linear', 'poly', 'rbf', 'sigmoid']

        for kernel in kernels:
            print(f'Modeling with {kernel} kernel')

            svcClf = SVC(kernel=kernel)
            svcClf.fit(self.Xtrain, self.yTrain)

            testAccuracy = svcClf.score(self.Xtest, self.yTest)

            self.svcStatistic[kernel] = {'test' : testAccuracy}
