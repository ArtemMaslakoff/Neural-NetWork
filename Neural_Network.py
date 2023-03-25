from ssl import ALERT_DESCRIPTION_DECOMPRESSION_FAILURE
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
    def __init__(self, teacher):
        self.maxNet = MaxNet()

        self.HemmingLay = []
        for i in range (0, len(teacher)):
            bias = 0
            self.HemmingLay.append(Neuron(teacher[i], bias, "X"))

    def feedForward(self, x):
        hemingLayOuts = []
        for i in range (0, len(self.HemmingLay)):
            hemingLayOuts.append(self.HemmingLay[i].feedforward(x))

        return self.maxNet.feedForward(hemingLayOuts)
    


class Perceptron:
    def __init__(self):
        weights = [1,0]
        bias = 0
        self.h1 = Neuron(weights, bias, "Sigmoid")

        weights = [1]
        bias = -0.5
        self.o1 = Neuron(weights, bias, "Or01")

    def feedForward(self, x):
        out_h1 = self.h1.feedforward(x)
        return self.o1.feedforward(out_h1)
        #return out_h1

    def teach(self, teacher, answers, stepLength, countOfSteps):
        for i in range (0, countOfSteps):
            print(self.h1.weights)
            for j in range (0, len(answers)):
                self.h1.weights[0] = self.h1.weights[0] - stepLength * self.h1.weights[0] * teacher[j][0]
                self.h1.weights[1] = self.h1.weights[1] - stepLength * self.h1.weights[0] * teacher[j][1]
                #self.h1.weights[1] = self.h1.weights[1] - stepLength * 2 * (self.h1.weights[0] * teacher[j][0] + self.h1.weights[1] * teacher[j][1] - answers[j]) * teacher[j][1]


## MAIN ##

#network = Comparator()
#x = np.array([34, 5])
#print(network.feedForward(x))


#maxnet = MaxNet()
#x = np.array([-12,0,-87,-23,-99,-1,-87])
#print(maxnet.feedForward(x));


#teacher = ([1,1,1,1,1,1],[1,-1,1,-1,-1,1])
#x = [-1,1,1,-1,1,1]
#hem = HemmingNet(teacher)
#print(hem.feedForward(x))



#teacher = ([0,0],
#           [0,1],
#           [1,0],
#           [1,1])
#answers = [1,1,0,0]
#per = Perceptron()
#per.teach(teacher, answers, 0.2, 50)
#x = [0,0]
#print(per.feedForward(x))
#x = [0,1]
#print(per.feedForward(x))
#x = [1,0]
#print(per.feedForward(x))
#x = [1,1]
#print(per.feedForward(x))

teacher = ([])