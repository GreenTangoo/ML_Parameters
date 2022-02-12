from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics



class DTreeClassifier:
    def __init__(self, Xtrain, Xtest, yTrain, yTest):
        self.Xtrain = Xtrain
        self.Xtest = Xtest
        self.yTrain = yTrain
        self.yTest = yTest

        self.dtStatistic = { 'best' : {}, 'random' : {}}

    def startModeling(self):
        print('Start modeling with DecisionTree model')

        self.testSplitter()

    def getJsonRepresentation(self):
        return self.dtStatistic

    def testSplitter(self):
        splitters = ['best', 'random']

        for splitter in splitters:
            self.testCriterion(splitter)

    def testCriterion(self, splitterType):
        criterions = ['gini', 'entropy']

        for criterion in criterions:
            self.testMaxDepth(splitterType, criterion)

    def testMaxDepth(self, splitterType, criterionType):
        maxDepthLst = range(2, 21)
        noneDepth = None

        depthStatistic = []

        for depth in maxDepthLst:
            testAccuracy = self.fitAndTestModel(splitterType, criterionType, depth)

            depthStatistic.append({str(depth) : { 'test' : testAccuracy }})


        testAccuracy = self.fitAndTestModel(splitterType, criterionType, noneDepth)

        depthStatistic.append({'21' : { 'test' : testAccuracy }})

        self.dtStatistic[splitterType][criterionType] = depthStatistic

    def fitAndTestModel(self, splitterType, criterionType, depth):
        print(f'Modeling with {splitterType} splitter | {criterionType} criterion | {depth} max depth level')

        dtClf = DecisionTreeClassifier(splitter=splitterType, criterion=criterionType, max_depth=depth)
        dtClf.fit(self.Xtrain, self.yTrain)

        return dtClf.score(self.Xtest, self.yTest)
