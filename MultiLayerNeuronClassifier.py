from sklearn.neural_network import MLPClassifier
from sklearn import metrics

class MultiLayerPerceptronClassifier:
    def __init__(self, Xtrain, Xtest, yTrain, yTest):
        self.Xtrain = Xtrain
        self.Xtest = Xtest
        self.yTrain = yTrain
        self.yTest = yTest

        self.mlpStatistic = { 'lbfgs' : {}, 'sgd' : {}, 'adam' : {} }


    def startModeling(self):
        print('Start modeling with MLPC model')

        self.testSolverType()

    def getJsonRepresentation(self):
        return self.mlpStatistic

    def testSolverType(self):
        solvers = ['lbfgs', 'sgd', 'adam']

        for solver in solvers:
            self.testActivationFunction(solver)


    def testActivationFunction(self, solverType):
        activations = ['identity', 'logistic', 'tanh', 'relu']

        for activationFunction in activations:
            self.testAmountNeuronsOnHiddenLayer(activationFunction, solverType)

    def testAmountNeuronsOnHiddenLayer(self, activationFunction, solverType):
        neuronsVariantsLst = [i for i in range(50, 110, 10)]
        neuronsStatistic = []

        for neurons in neuronsVariantsLst:
            print(f'Modeling with {solverType} solver type | {activationFunction} activation function | {neurons} amount neurons')

            mlpcClf = MLPClassifier(solver=solverType, activation=activationFunction, hidden_layer_sizes=(neurons,))
            mlpcClf.fit(self.Xtrain, self.yTrain)

            testAccuracy = mlpcClf.score(self.Xtest, self.yTest)

            neuronsStatistic.append({ neurons : { 'test' : testAccuracy } } )

        self.mlpStatistic[solverType][activationFunction] = neuronsStatistic
