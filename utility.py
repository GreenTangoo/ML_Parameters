DICT_INDEX = 1
KEY_INDEX = 0


def findKeyByBestTestAccuracy(statistic):
    global TUPLE_INDEX
    global KEY_INDEX

    maxItem = max(statistic.items(), key=lambda x:x[DICT_INDEX]['test'])
    return maxItem[KEY_INDEX]

def getDataForPlotKnnModel(jsonContainer):
    testAccuracy = []
    neighbors = []

    for algType in jsonContainer:
        for amountNeighbors in jsonContainer[algType]:
            neighbors.append(amountNeighbors)

            for distanceAlgType in jsonContainer[algType][amountNeighbors]:
                for accuracy in jsonContainer[algType][amountNeighbors][distanceAlgType]:
                    testAccuracy.append(jsonContainer[algType][amountNeighbors][distanceAlgType][accuracy])

    return neighbors, testAccuracy


def getDataForLgsModel(jsonContainer, penaltyType):
    testAccuracy = []
    solvers = []

    for algType in jsonContainer:
        penaltyJsonContainer = jsonContainer[algType][penaltyType]

        for solverContainer in penaltyJsonContainer:
            solver = list(solverContainer.keys())[0]
            solvers.append(solver)

            testAccuracy.append(solverContainer[solver]['test'])

    return solvers, testAccuracy


def getDataForDtModel(jsonContainer, splitter, criterion):
    testAccuracy = []
    depths = []

    for algType in jsonContainer:
        splitterCriterionContainer = jsonContainer[algType][splitter][criterion]

        for oneDepthContainer in splitterCriterionContainer:
            for depth in oneDepthContainer:
                depths.append(depth)
                testAccuracy.append(oneDepthContainer[depth]['test'])

    return depths, testAccuracy


def getDataForSvcModel(jsonContainer):
    testAccuracy = []
    kernels = []

    for algType in jsonContainer:
        kernelsContainer = jsonContainer[algType]

        for kernelKey in kernelsContainer:
            kernels.append(kernelKey)
            testAccuracy.append(kernelsContainer[kernelKey]['test'])


    return kernels, testAccuracy


def getDataForMlpcModel(jsonContainer, solverType, activationType):
    testAccuracy = []
    neurons = []

    for algType in jsonContainer:
        solversContainer = jsonContainer[algType]

        neuronsContainers = solversContainer[solverType][activationType]

        for neuronContainer in neuronsContainers:
            for neuronKey in neuronContainer:
                neurons.append(neuronKey)
                testAccuracy.append(neuronContainer[neuronKey]['test'])

    return neurons, testAccuracy
