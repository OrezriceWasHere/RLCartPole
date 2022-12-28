import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class PolicyGradientAgent(object):
    def __init__(self, env, lr=0.1):
        self.num_actions = env.action_space.n
        self.num_features = env.observation_space.shape[0]
        self.W = np.zeros((self.num_features, self.num_actions))
        self.b = np.zeros((self.num_actions))
        self.lr = lr

    def action_probability(self, state):
        '''
        Compute p(a|s) for discrete action using linear model on state
        :param state: environment state
        :return: vector of probabilities
        '''
        scores = np.asarray(state).dot(self.W) + self.b
        probabilities = softmax(scores)
        return probabilities
        #TODO

    def get_action(self, state):
        '''
        Selects a random action according to p(a|s)
        :param state: environment state
        :return: action
        '''
        probs = self.action_probability(state)
        return np.random.choice(self.num_actions, p=probs)

    def grad_log_prob(self, state, action):
        '''
        Compute gradient of log P(a|S) w.r.t W and b
        :param state: environment state
        :param action: descrete action taken
        :return: dlogP(a|s)/dW, dlogP(a|s)/db
        '''
        action_chosen_one_hot = np.zeros(2)
        action_chosen_one_hot[action] = 1
        gradient_b = action_chosen_one_hot - self.action_probability(state)
        gradient_w = np.outer(state, gradient_b)
        return gradient_w, gradient_b

        # TODO

    def update_weights(self, dW, db):
        '''
        Updates weights using simple gradient ascent
        :param dW: gradients w.r.t W
        :param db: gradients w.r.t b
        '''
        self.W += self.lr * dW
        self.b += self.lr * db


