import numpy
import theano
import theano.tensor as T
import pickle
import cPickle
from Layers.RNNHiddenLayer import *
from Layers.SoftmaxLayer import *
from Utils.CostFHelper import *

class RNN:
    def __init__(self,
                 rng,
                 numIn,
                 numHidden,
                 truncate,
                 useSoftmax = False,
                 activation = T.tanh,
                 learningRate = 0.005):
        # Set parameters
        self.Rng = rng
        self.NumIn = numIn
        self.NumHidden = numHidden
        self.Truncate = truncate
        self.UseSoftmax = useSoftmax
        self.Activation = activation
        self.LearningRate = learningRate

        self.createRNN()
        self.buildRNN()

    def createRNN(self):
        # Save shared parameters
        self.Whh = None
        self.Wx = None
        self.Wy = None

        # Create RNN model
        self.Layers = []
        for layerId in range(self.Truncate):
            if layerId == 0:
                hiddenLayer = RNNHiddenLayer(
                    rng=self.Rng,
                    numIn=self.NumIn,
                    numHidden=self.NumHidden,
                    activation=self.Activation
                )
                self.Whh = hiddenLayer.Whh
                self.Wx = hiddenLayer.Wx
                self.Wy = hiddenLayer.Wy
            else:
                hiddenLayer = RNNHiddenLayer(
                    rng=self.Rng,
                    numIn=self.NumIn,
                    numHidden=self.NumHidden,
                    activation=self.Activation,
                    whh=self.Whh,
                    wx=self.Wx,
                    wy=self.Wy
                )
            self.Layers.append(hiddenLayer)

    def buildRNN(self):
        # Create train model
        X = T.matrix('X', dtype = theano.config.floatX)
        Y = T.matrix('Y', dtype = theano.config.floatX)

        # Feed-forward
        S = numpy.zeros((1, self.NumHidden), dtype = theano.config.floatX)
        for idx, layer in enumerate(self.Layers):
            [S, Yp] = layer.FeedForward(S, X[idx])

        # Calculate cost function
        softmaxLayer = SoftmaxLayer(Yp)
        Yp = softmaxLayer.Output()
        cost = CategoryEntropy(Yp, Y)

        # Calculate error function

        # Get params and calculate gradients
        params = self.Layers[-1].Params
        grads = T.grad(cost, params)
        updates = [(param, param - self.LearningRate * grad)
                   for (param, grad) in zip(params, grads)]

        self.TrainModel = theano.function(
            inputs = [X, Y],
            outputs = cost,
            updates = updates,
        )

        # Create test model
        # self.TestModel = theano.function(
        #     inputs=[X, Y],
        #     outputs=rnnModel.Cost(Y),
        #     updates=rnnModel.Updates()
        # )

    def Output(self):
        output = self.Layers[-1].Output
        if self.UseSoftmax is False:
            return output
        else:
            softmaxLayer = SoftmaxLayer(
                input = output
            )
            return softmaxLayer.Output()



    def LoadModel(self, fileName):
        file = open(fileName, 'r')
        self.Whh.set_value(cPickle.load(file), borrow = True)
        self.Whh.set_value(cPickle.load(file), borrow=True)
        self.Whh.set_value(cPickle.load(file), borrow=True)
        file.close()

    def SaveModel(self, fileName):
        file = open(fileName, 'wb')
        pickle.dump(self.Whh.get_value(borrow = True), file, -1)
        pickle.dump(self.Wx.get_value(borrow = True), file, -1)
        pickle.dump(self.Wy.get_value(borrow = True), file, -1)
        file.close()

