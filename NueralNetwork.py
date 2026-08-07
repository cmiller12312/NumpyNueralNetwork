import numpy
from dataset import Dataset
from sklearn.neural_network import MLPClassifier

class NueralNetwork:
    def __init__(self, epoch=5000, lossFunction="MSE", learningRate=0.01, layers=None, hiddenLayerActivation="ReLU", OutputActivation="sigmoid", convergenceMargin=0.001) -> None:
        #activation is RELU by default
        #iteration count of none will run it automatically until no chnages or in a loop
        #loss function is Mean sqaured error by default
        #layers will be a list of perceptrons counts per layer
        self.hiddenLayerActivation = hiddenLayerActivation
        self.OutputActivation = OutputActivation
        self.epoch = epoch
        self.lossFunction =lossFunction
        self.learningRate = learningRate
        self.weights = []
        self.biases = []
        self.convergenceMargin = convergenceMargin
        

        #z is the pre activation value needed later for the chain rule
        self.zs = []

        #this is the post activation value for each later needed for chain rule
        self.activations = []

        #this is just a flag to make sure its set up correctly
        self.status = False

        if layers == None:
            print("Please define layers")
            return

        rng = numpy.random.default_rng(42)
        for i in range(len(layers)):
            try:
                int(layers[i])
                if i == 0:
                    continue
                #weights is num of perceptrons by num of inputs
                #no special tools for setting inputs weights to 0
                #for now sets all weights to random
                limit = numpy.sqrt(2 / layers[i-1])

                self.weights.append(rng.normal(0,limit,(layers[i], layers[i-1])))
                #just weights for bias
                self.biases.append(numpy.zeros((layers[i], 1)))
            except:
                print("Failed to make network. Make sure layer counts are defined only as integers. Reminder layers defines count for both hidden and non hidden")
                return
        self.status = True
            
    def forwardFeed(self, input):
            self.zs = []
            self.activations = [input]

            count = 0 
            while True:
                temp = (numpy.dot(self.weights[count], input)) + self.biases[count]
                self.zs.append(temp)
                if count == len(self.weights) - 1:
                    temp = self.activationFunction(temp, self.OutputActivation, False)
                    self.activations.append(temp)
                    return temp
                else:
                    temp = self.activationFunction(temp, self.hiddenLayerActivation, False)
                    self.activations.append(temp)
                    input = temp
                    count += 1

    

    def trainSingle(self, input, actual):
        input = numpy.array(input).reshape(-1, 1)
        actual = numpy.array(actual).reshape(-1, 1)

        prediction = self.forwardFeed(input)

        loss = self.runLossSingle(prediction, actual, self.lossFunction)

        self.backwardFeed(actual)

    def runLossSingle(self, predicted, actual, method):
        if method == "MSE":
            return self.MSElossSingle(predicted, actual)
        if method == "crossEntropy":
            return self.crossEntropy(predicted, actual)

    def trainSingles(self, input, actual):

        #this had automatic convergence spotting
        patience = 0 
        lastLoss = float("inf")
        temp = 0 
        for epoch in range(self.epoch):
        
                totalLoss = 0
        
                indices = numpy.random.permutation(len(input))
        
                for i in indices:
                    self.trainSingle(input[i], actual[i])
        
                for i in range(len(input)):
                    prediction = self.forwardFeed(input[i].reshape(-1,1))
                    totalLoss += self.runLossSingle(
                        prediction,
                        actual[i].reshape(-1,1),
                        self.lossFunction
                    )

                averageLoss = totalLoss / len(input)
                if numpy.abs(lastLoss - averageLoss) <= self.convergenceMargin:
                    patience += 1
                    print("patience: ", patience)
                else:
                    patience = 0

                lastLoss = averageLoss
                if patience >= 10:
                    break
                print(epoch)

    def MSElossSingle(self, predicted, actual):
        # we multiply by 0.5 because when we take the derivative and get 2(output - actual) it cancels out 
        predicted = numpy.clip(predicted, 1e-12, 1)
        return numpy.mean(0.5 * numpy.square(predicted - actual))

    def crossEntropy(self, predicted, actual):

        #clipping will prevent the value from skewing it too hard
        predicted = numpy.clip(predicted, 1e-12, 1)
        return -numpy.sum(actual * numpy.log(predicted))


    def activationFunction(self, input, type, derivative):
        if type == "ReLU":
            return self.ReLU(input, derivative=derivative)
        if type == "sigmoid":
            return self.sigmoid(input, derivative=derivative)
        if type == "softmax":
            return self.softmax(input)
    

    def softmax(self, input):
        expValues = numpy.exp(input - numpy.max(input))
        return expValues / numpy.sum(expValues)
    
    def ReLU(self, input, derivative=False):
        if derivative:
            return (input > 0).astype(float)
        return numpy.maximum(0, input)
        
    def sigmoid(self, input, derivative=False):
        input = numpy.clip(input, -500, 500)
        if derivative:
            s = 1 / (1 + numpy.exp(-input))
            return s * (1 - s)

        return 1 / (1 + numpy.exp(-input))

    def backwardFeed(self, actual):
        numOfLayers = len(self.weights)

        weightGradients = [None] * numOfLayers
        biasGradients = [None] * numOfLayers

        

        #getting the output layer
        output = self.activations[-1]
        #dL/dZ = dL/activation * activation/dz

        #chain rule applies here
        #since output is defined by me to have a possible different activation we do this. 
        #WHEN ADDING MORE THAN MSE THIS STEP WILL NEED TO CHANGE (output - actual) is the derivative only for 1/2 * mse
        if self.OutputActivation == "softmax" and self.lossFunction == "crossEntropy":
            #soft max and cross entrop cnacel each other out which is why only dZ is here
            delta = output - actual

        elif self.OutputActivation == "sigmoid" and self.lossFunction == "MSE":
            delta = (output - actual) * self.sigmoid(self.zs[-1], derivative=True)

        else:
            raise Exception("Unsupported activation/loss combination")


        # dL/dW = dZ/dW * dz/dZ * dL/da
        # here its only delta * dW because delta is already defined as dA/dZ * dL/dA 
        weightGradients[-1] = numpy.dot(delta, self.activations[-2].T)

        #bias is just one so dL/dB = dZ/dB * da/dZ * dL/da translates to da/dZ * dL/da which delta already defines
        biasGradients[-1] = delta

        #stepping down and stopping at -1
        #starting at -2 because we already did output layer and -1 for lists
        for i in range(numOfLayers - 2, -1, -1):

            delta = numpy.dot(self.weights[i + 1].T, delta) * self.activationFunction(self.zs[i], self.hiddenLayerActivation, True) 
        
            #same as output layer logic
            weightGradients[i] = numpy.dot(delta, self.activations[i].T )
            biasGradients[i] = delta

        #to update gradients we apply to gradients to the to the weights times the learning rate so we get
        for i in range(numOfLayers):
            #prevents extra bad run to alter it too much
            weightGradients[i] = numpy.clip(weightGradients[i], -1, 1)
            biasGradients[i] = numpy.clip(biasGradients[i], -1, 1)

            self.weights[i] -= self.learningRate * weightGradients[i]
            self.biases[i] -= self.learningRate * biasGradients[i]

if __name__ == "__main__":
    X, y = Dataset.loadDigits()

    X = numpy.array(X, dtype=float)
    y = numpy.array(y)

    numClasses = len(numpy.unique(y))

    std = numpy.std(X, axis=0)

    std[std == 0] = 1

    X = (X - numpy.mean(X, axis=0)) / std
    nn = NueralNetwork(
        layers=[X.shape[1],64,32, numClasses],
        learningRate=0.01,
        OutputActivation="softmax",
        lossFunction="crossEntropy",
        convergenceMargin=1e-5
    )

    indices = numpy.random.permutation(len(X))

    split = int(0.7 * len(X))

    trainIndices = indices[:split]
    testIndices = indices[split:]

    xTrain = X[trainIndices]
    yTrain = y[trainIndices]
    yTrainEncoded = numpy.eye(numClasses)[yTrain]

    xTest = X[testIndices]
    yTest = y[testIndices]
    yTestEncoded = numpy.eye(numClasses)[yTest]

    print("Train:", xTrain.shape, yTrain.shape)
    print("Test:", xTest.shape, yTest.shape)
    
    nn.trainSingles(xTrain, yTrainEncoded)


    correct = 0
    totalLoss = 0

    for i in range(len(xTest)):
        prediction = nn.forwardFeed(
            xTest[i].reshape(-1,1)
        )

        actual = yTest[i]

        # calculate loss
        totalLoss += nn.runLossSingle(
            prediction,
            yTestEncoded[i].reshape(-1,1),
            nn.lossFunction
        )

        predictedClass = numpy.argmax(prediction)

        if predictedClass == actual:
            correct += 1

        print(predictedClass, " ---- ", actual)

        

    accuracy = correct / len(xTest)
    averageLoss = totalLoss / len(xTest)

    print("\n======== TEST RESULTS ========")
    print("Samples:", len(xTest))
    print("Accuracy:", accuracy)
    print("Average Loss:", averageLoss)

    clf = MLPClassifier(
        hidden_layer_sizes=(64,64),
        activation="relu",
        solver="sgd",
        learning_rate="constant",
        learning_rate_init=0.01,
        batch_size=1,
        momentum=0,
        max_iter=5000,
        random_state=42
    )

    clf.fit(xTrain, yTrain)

    print("SkiKit: ", clf.score(xTest, yTest))

    #print(nn.forwardFeed(numpy.matrix([[1], [4], [5], [6]])))
