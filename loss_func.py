import numpy as np


def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    # return a pair metric_name, result
    # since preds are margin(before logistic transformation, cutoff at 0)

    return 'error', float(sum(labels != (preds > 0.0))) / len(labels)


class Focal_Binary_Loss:
    '''
    The class of focal loss, allows the users to change the gamma parameter
    '''

    def __init__(self, gamma_indct):
        '''
        :param gamma_indct: The parameter to specify the gamma indicator
        '''
        self.gamma_indct = gamma_indct

    def robust_pow(self, num_base, num_pow):
        # numpy does not permit negative numbers to fractional power
        # use this to perform the power algorithmic

        return np.sign(num_base) * (np.abs(num_base)) ** (num_pow)

    def focal_binary_object(self, pred, dtrain):
        gamma_indct = self.gamma_indct
        # retrieve data from dtrain matrix
        label = dtrain.get_label()
        # compute the prediction with sigmoid
        sigmoid_pred = 1.0 / (1.0 + np.exp(-pred))
        # gradient
        # complex gradient with different parts
        g1 = sigmoid_pred * (1 - sigmoid_pred)
        g2 = label + ((-1) ** label) * sigmoid_pred
        g3 = sigmoid_pred + label - 1
        g4 = 1 - label - ((-1) ** label) * sigmoid_pred
        g5 = label + ((-1) ** label) * sigmoid_pred
        # combine the gradient
        grad = gamma_indct * g3 * self.robust_pow(g2, gamma_indct) * np.log(g4 + 1e-9) + \
               ((-1) ** label) * self.robust_pow(g5, (gamma_indct + 1))
        # combine the gradient parts to get hessian components
        hess_1 = self.robust_pow(g2, gamma_indct) + \
                 gamma_indct * ((-1) ** label) * g3 * self.robust_pow(g2, (gamma_indct - 1))
        hess_2 = ((-1) ** label) * g3 * self.robust_pow(g2, gamma_indct) / g4
        # get the final 2nd order derivative
        hess = ((hess_1 * np.log(g4 + 1e-9) - hess_2) * gamma_indct +
                (gamma_indct + 1) * self.robust_pow(g5, gamma_indct)) * g1

        return grad, hess


class Weight_Binary_Cross_Entropy:
    '''
    The class of binary cross entropy loss, allows the users to change the weight parameter
    '''

    def __init__(self, imbalance_alpha):
        '''
        :param imbalance_alpha: the imbalanced \alpha value for the minority class (label as '1')
        '''
        self.imbalance_alpha = imbalance_alpha

    def weighted_binary_cross_entropy(self, pred, dtrain):
        # assign the value of imbalanced alpha
        imbalance_alpha = self.imbalance_alpha
        # retrieve data from dtrain matrix
        label = dtrain.get_label()
        # compute the prediction with sigmoid
        sigmoid_pred = 1.0 / (1.0 + np.exp(-pred))
        # gradient
        grad = -(imbalance_alpha ** label) * (label - sigmoid_pred)
        hess = (imbalance_alpha ** label) * sigmoid_pred * (1.0 - sigmoid_pred)

        return grad, hess