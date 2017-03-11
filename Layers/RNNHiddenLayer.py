import theano
import theano.tensor as T
import numpy

class RNNHiddenLayer:


    def __init__(self,
                 rng,
                 numIn,
                 numHidden,
                 whh = None,
                 wx = None,
                 wy = None,
                 sActivation = T.tanh,
                 yActivation = None):
        # Set parameters
        self.Rng = rng
        self.NumIn = numIn
        self.NumHidden = numHidden
        self.Whh = whh
        self.Wx = wx
        self.Wy = wy
        self.SActivation = sActivation
        self.YActivation = yActivation

        self.createModel()

    def createModel(self):
        if self.Whh is None:
            wBound = numpy.sqrt(6.0 / (self.NumHidden + self.NumHidden))
            self.Whh = theano.shared(
                numpy.asarray(self.Rng.uniform(
                        low  = -wBound,
                        high =  wBound,
                        size = (self.NumHidden, self.NumHidden)
                    ),
                    dtype = theano.config.floatX
                ),
                borrow = True
            )

        if self.Wx is None:
            wBound = numpy.sqrt(6.0 / (self.NumIn + self.NumHidden))
            self.Wx = theano.shared(
                numpy.asarray(self.Rng.uniform(
                        low  = -wBound,
                        high =  wBound,
                        size = (self.NumIn, self.NumHidden)
                    ),
                    dtype = theano.config.floatX
                ),
                borrow = True
            )

        if self.Wy is None:
            wBound = numpy.sqrt(6.0 / (self.NumHidden + self.NumIn))
            self.Wy = theano.shared(
                numpy.asarray(self.Rng.uniform(
                        low  = -wBound,
                        high =  wBound,
                        size = (self.NumHidden, self.NumIn)
                    ),
                    dtype = theano.config.floatX
                ),
                borrow = True
            )

        self.Params = [self.Whh, self.Wx, self.Wy]

    def FeedForward(self, Skm1, Xk):
        S = self.SActivation(self.Wx[Xk] + T.dot(Skm1, self.Whh))
        if self.YActivation is None:
            Y = T.dot(S, self.Wy)
        else:
            Y = self.YActivation(T.dot(S, self.Wy))
        return [S, Y]

