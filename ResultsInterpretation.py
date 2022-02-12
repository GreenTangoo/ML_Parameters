import matplotlib.pyplot as plt
import json

import utility

class PlotsCreator:
    def __init__(self):
        self.creatorsDict = {
            'knn' : self.__createKnnPlot,
            'lgs' : self.__createLgsPlot,
            'dt' : self.__createDtPlot,
            'svc' : self.__createSvcPlot,
            'mlpc' : self.__createMlpcPlot
        }

    def createPlot(self, jsonContainer, algorithmType):
        self.creatorsDict[algorithmType](jsonContainer)

    def __createKnnPlot(self, jsonContainer):
        neighbors, testAccuracy = utility.getDataForPlotKnnModel(jsonContainer)

        plt.plot(neighbors, testAccuracy)

        plt.xlabel('Amount neighbors')
        plt.ylabel('Test accuracy')

        plt.show()

    def __createLgsPlot(self, jsonContainer):
        penaltyTypes = ['none', 'l1', 'l2']

        for penalty in penaltyTypes:
            solvers, testAccuracy = utility.getDataForLgsModel(jsonContainer, penalty)

            plt.plot(solvers, testAccuracy)

            plt.title(penalty)

            plt.xlabel('Solvers')
            plt.ylabel('Test accuracy')

            plt.show()

    def __createDtPlot(self, jsonContainer):
        splitterTypes = ['best', 'random']
        criterionTypes = ['gini', 'entropy']

        for splitter in splitterTypes:
            for criterion in criterionTypes:
                depths, testAccuracy = utility.getDataForDtModel(jsonContainer, splitter, criterion)

                plt.plot(depths, testAccuracy)

                plt.title(f'Splitter = {splitter} | Sriterion = {criterion}')

                plt.xlabel('Depth')
                plt.ylabel('Test accuracy')

                plt.show()

    def __createSvcPlot(self, jsonContainer):
        kernels, testAccuracy = utility.getDataForSvcModel(jsonContainer)

        plt.plot(kernels, testAccuracy)

        plt.xlabel('Kernel')
        plt.ylabel('Test accuracy')

        plt.show()


    def __createMlpcPlot(self, jsonContainer):
        solverTypes = ['lbfgs', 'sgd', 'adam']
        activationTypes = ['identity', 'logistic', 'tanh', 'relu']

        for solver in solverTypes:
            for activation in activationTypes:
                neurons, testAccuracy = utility.getDataForMlpcModel(jsonContainer, solver, activation)

                plt.plot(neurons, testAccuracy)

                plt.title(f'Solver = {solver} | Activation function = {activation}')

                plt.xlabel('Amount neurons in hidden layer')
                plt.ylabel('Test accuracy')

                plt.show()


class JsonWriter:
    def __init__(self):
        pass

    def write(self, jsonContainer, setThreshold):
        algorithmType = list(jsonContainer.keys())[0]

        filename = algorithmType + '_' + str(setThreshold) + '_Result.json'
        filepath = 'Results/' + filename

        with open(filepath, 'w') as file:
            json.dump(jsonContainer, file, indent=4)
