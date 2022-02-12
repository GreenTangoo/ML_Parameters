from sklearn.linear_model import LogisticRegression
from sklearn import metrics



class LogistiRegressionClassifier:
    def __init__(self, Xtrain, Xtest, yTrain, yTest):
        self.Xtrain = Xtrain
        self.Xtest = Xtest
        self.yTrain = yTrain
        self.yTest = yTest

        self.lgsStatistic = {}

    def setBadQueriesAmount(self, amountBadQueries):
        self.badCount = amountBadQueries

    def setValidQueriesAmount(self, amountValidQueries):
        self.validCount = amountValidQueries

    def startModeling(self):
        print('Start modeling with LogisticRegression model')
        self.testPenaltyParameter()

    def getJsonRepresentation(self):
        return self.lgsStatistic

    def testPenaltyParameter(self):
        penaltyTypes = ['none', 'l1', 'l2']

        for penalty in penaltyTypes:
            self.testSolverParameter(penalty)

    def testSolverParameter(self, penaltyType):
        solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        solversStatistic = []

        for solver in solvers:
            print(f'Modeliing with {penaltyType} penalty type | {solver} solver type')

            try:
                lgs = LogisticRegression(class_weight={1: 2 * self.validCount / self.badCount, 0: 1.0}, penalty=penaltyType, solver=solver)
                lgs.fit(self.Xtrain, self.yTrain)

                yPred = lgs.predict(self.Xtest)
                testAccuracy = lgs.score(self.Xtest, self.yTest)

                solversStatistic.append({solver : {'test' : testAccuracy}})
            except ValueError:
                print(f'Cannot fit model with {penaltyType} penalty and {solver} solver')
                continue

        self.lgsStatistic[penaltyType] = solversStatistic
