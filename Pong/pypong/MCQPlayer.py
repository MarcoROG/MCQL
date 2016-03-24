from MCQLearner import MCQLearner

class MCQPlayer(object):
    def __init__(self):
        self.learner = MCQLearner(4,3,0.001,0.9,0.65)
        self.lastReward = None
        self.lastQ = None
        self.lastAction = None
        self.lastState = None
        
    def update(self, paddle, game):
        resp = None
        state = None
        state = [game.ball.rect.x, game.ball.rect.y, paddle.rect.x, paddle.rect.y]
        if self.lastReward == None:
            resp = self.learner.step(state)
        else:
            resp = self.learner.stepRew(state, self.lastQ, self.lastReward)
        #print str(resp[0]) + "-" + str(resp[1])
        if resp[0] == 0:
            paddle.direction = 1
        elif resp[0] == 1:
            paddle.direction = 0
        else : 
            paddle.direction = -1
        self.lastState = state
        self.lastQ = resp[1]
        self.lastAction = resp[0]
        self.lastReward = None

    def hit(self):
        #self.lastReward = 1
        #self.learner.stepRew(self.lastState, self.lastQ, -8)
        pass
    def lost(self):
        #self.lastReward = -10
        self.learner.stepRew(self.lastState, self.lastQ, 0)
        print self.lastQ
        
    def won(self):
        #self.lastReward = 5
        self.learner.stepRew(self.lastState, self.lastQ, 10)


