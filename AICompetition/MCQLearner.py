import theano
import theano.tensor as T
import numpy as np
 
class MCQLearner:
    ### Implements page 8 figure 1 algorithm from 1994 ON-LINE Q-LEARNING USING CONNECTIONIST SYSTEMS G. A. Rummery & M. Niranjan
    ## Todo: Implement epsilon greedy policy
    def __init__(self, stateSize, actionCount, lr, df, lt):
        stateSize =             stateSize + 1
        max =                   np.sqrt(6. / (stateSize + actionCount))
        min =                   -np.sqrt(6. / (stateSize + actionCount))
        self.learningRate =     lr
        self.discountFactor =   df
        self.lambdaTraces =     lt
 
        W_values =              np.random.rand(stateSize, stateSize) * (max - min) + min
        self.W =                theano.shared(value = W_values, name = 'W', borrow = False)
        W_valuesT =              np.random.rand(stateSize, actionCount) * (max - min) + min
        self.WT =               theano.shared(value = W_valuesT, name = 'WT', borrow = False)
 
        e_values =              np.zeros_like(W_values)
        self.e =                theano.shared(value = e_values, name = 'e', borrow = False)
        eT_values =             np.zeros_like(W_valuesT)
        self.eT =               theano.shared(value = eT_values, name = 'eT', borrow = False)
 
        state =                 T.fvector('state')
        learningRate =          T.fscalar('lr')
        discountFactor =        T.fscalar('y')
        lambdaTraces =          T.fscalar('lambdaTraces')
        oldReward =             T.fscalar('reward')
        oldQ =                  T.fscalar('oldQ')
 
        self.Qvalues =          T.dot(T.nnet.sigmoid(T.dot(state,self.W)),self.WT)
        self.bestAction =       T.argmax(self.Qvalues)
        self.bestQ =            T.max(self.Qvalues)
 
        self.dQdW =             T.grad(self.bestQ, self.W)
        self.dQdWT =            T.grad(self.bestQ, self.WT)
 
        update = []
        W_update =              (self.W, self.W + learningRate*(oldReward + discountFactor*self.bestQ - oldQ)*self.e )
        WT_update =             (self.WT, self.WT + learningRate*(oldReward + discountFactor*self.bestQ - oldQ)*self.eT )
        e_update =              (self.e, self.dQdW + lambdaTraces*discountFactor*self.e)
        eT_update =             (self.eT, self.dQdWT + lambdaTraces*discountFactor*self.eT)
        update.append(W_update)
        update.append(e_update)
        update.append(WT_update)
        update.append(eT_update)

        self.deceide =          theano.function([state], [self.bestAction, self.bestQ], allow_input_downcast=True)
        self.deceideUpdate =    theano.function([state, oldQ, oldReward, learningRate, discountFactor, lambdaTraces], [self.bestAction, self.bestQ], updates = update, allow_input_downcast=True)
 
    def stepRew(self, state, oldQ, oldReward):
        return self.deceideUpdate(np.insert(state, 0, 1), oldQ, oldReward, self.learningRate, self.discountFactor, self.lambdaTraces)
    def step(self, state):
        return self.deceide(np.insert(state, 0, 1))