from email import feedparser
import numpy as np

class Neuron:
    def __init__(self, weights, bias, function):
        self.weights = weights
        self.bias = bias
        self.function = function

    def Sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def X(self, x):
        return x

    def Or01(self, x):
        if x < 0:
            return 0
        else:
            return 1

    def Or11(self, x):
        if x < 0:
            return -1
        else:
            return 1

    def Or0X(self, x):
        if x < 0:
            return 0
        else:
            return x

    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        if(self.function == "Sigmoid"):
            return self.Sigmoid(total)
        if(self.function == "X"):
            return self.X(total)
        if(self.function == "Or01"):
            return self.Or01(total)
        if(self.function == "Or11"):
            return self.Or11(total)
        if(self.function == "Or0X"):
            return self.Or0X(total)

   

class Comparator:
    def __init__(self):

        weights = np.array([1,-1])
        bias = 0
        self.h1 = Neuron(weights, bias, "Or0X")

        weights = np.array([-1,1])
        bias = 0
        self.h2 = Neuron(weights, bias, "Or0X")

        weights = np.array([0.5,0.5,0.5,0.5])
        bias = 0
        self.o1 = Neuron(weights, bias, "X")

    def feedForward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)

        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2, x[0], x[1]]))

        return out_o1



class MaxNet:
    def __init__(self):
        self.comparator = Comparator()

        weights = np.array([-1,1])
        bias = 0
        self.h1 = Neuron(weights, bias, "Or01")

    def feedForward(self, x):
        out_o1 = x[0];
        for i in range (len(x) - 1):
            out_o1 = self.comparator.feedForward([out_o1, x[i+1]])

        result = []
        for i in range (0, len(x)):
            result.append(self.h1.feedforward([out_o1, x[i]]))
        return [out_o1, result]



class HemmingNet:
    def __init__(self):
        self.maxNet = MaxNet()

        weights = np.array([-1,1])
        bias = 0
        self.h1 = Neuron(weights, bias, "Or01")

    def feedForward(self, x):
        out_o1 = x[0];
        for i in range (len(x) - 1):
            out_o1 = self.comparator.feedForward([out_o1, x[i+1]])

        result = []
        for i in range (0, len(x)):
            result.append(self.h1.feedforward([out_o1, x[i]]))
        return [out_o1, result]



#class NeuralNetwork:
#    def __init__(self):
#        weights = np.array([0,1])
#        bias = 0

#        self.h1 = Neuron(weights, bias, "Sigmoid")
#        self.h2 = Neuron(weights, bias, "Sigmoid")
#        self.o1 = Neuron(weights, bias, "Sigmoid")

#    def feedForward(self, x):
#        out_h1 = self.h1.feedforward(x)
#        out_h2 = self.h2.feedforward(x)

#        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))

#        return out_o1



#class Perceptron:
#    def __init__(self, ):
#        weights = np.array([1,1,1,1])
#        bias = 0
#        self.h1 = Neuron(weights, bias, "Or01")

#    def feedForward(self, x):
#        out_h1 = self.h1.feedforward(x)

#        out_o1 = self.o1.feedforward([out_h1,out_h2,out_h3,out_h4,out_h5,out_h6])
#        return out_o1
    
    #def CalculateError(self):

    #def Teach(self, countOfEpochs, teacher, answers):
    #    for i in range(countOfEpochs):
    #        error = 0
    #        for j in range(len(teacher)):
    #            error += self.feedForward(teacher[j]) - answers[j]
    #        print(error)
    #        self.ChangeWeights(error)




## MAIN ##

network = Comparator()
x = np.array([34, 5])
print(network.feedForward(x))


maxnet = MaxNet()
x = np.array([-12,0,87,-23,-99,-1,87])
print(maxnet.feedForward(x));


#teacher = ([0,0,0,0],[0,1,0,0],[1,0,0,0],[1,1,0,0],
#           [0,0,0,1],[0,1,0,1],[1,0,0,1],[1,1,0,1],
#           [0,0,1,0],[0,1,1,0],[1,0,1,0],[1,1,1,0],
#           [0,0,1,1],[0,1,1,1],[1,0,1,1],[1,1,1,1])
#answers = ([0,0,0,1,0,1,1,0,0,1,1,0,1,0,0,0])
#perc = Perceptron()
#perc.Teach(10, teacher, answers);