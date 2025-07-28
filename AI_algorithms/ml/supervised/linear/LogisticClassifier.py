from ..base import GradientDescentSupervisedModel
import numpy as np # type: ignore

def sigma(x):
    """
    Sigmoid function.
    
    :param x: Input value
    :return: Sigmoid of x
    """
    return 1 / (1 + np.exp(-x))

class LogisticClassifier(GradientDescentSupervisedModel):
    """
    Implementation of a linear logistic classifier using gradient descent.
    
    This class provides methods for training and predicting using the logistic model.
    """

    def __init__(self, w_init=None):
        """
        Initialize the LogisticClassifier with a name.
        
        :param name: Name of the classifier
        """
        super().__init__(name='Logistic Classifier')
        self.w = w_init.astype(float) if w_init is not None else None
        if self.w is None:
            self.ndim = None
    
    def __fw(self, x):
        """
        w: numpy array of shape (n+1,) w[1:] coefficients of x, w[0] independent
            term
        x: numpy array of shape (n,)
        returns:
        a scalar y, with the linear combination of x and w
        """
        return x @ self.w[1:] + self.w[0]
        

    def f(self,x):
        """
        w: numpy array of shape (n+1,) w[1:] coefficients of x, w[0] independent
            term
        x: numpy array of shape (n,)
        returns:
        a scalar y, with y<0.5 for class 0, y>=0.5 for class 1
        """
        return sigma(self.__fw(x))
    
    def init_params(self, random_state=None):
        """
        Initialize the model parameters.
        
        This method initializes the weights of the model. If the weights are not
        provided, they are randomly initialized.
        """
        if random_state is not None:
            np.random.seed(random_state)
        if self.w is None:
            self.w = np.random.rand(self.ndim + 1)

    def update_params(self, delta, lr=0.01):
        """
        Update the model parameters using the gradient.
        
        :param delta: Gradient
        :param lr: Learning rate
        """
        self.w -= lr * delta

    def loss(self, X, y):
        """
        Compute the loss function.
        
        :param X: Data
        :param y: Target values
        :return: Loss
        """
        return -np.sum(y * np.log(self.f(X)) + (1 - y) * np.log(1 - self.f(X))) / len(X)

    def grad(self, X, y):
        """
        Compute the gradient of the loss function.
        
        :param X: Data
        :param y: Target values
        :return: Gradient
        """
        x_prime = np.zeros(len(X) + 1)
        x_prime[1:] = X
        x_prime[0] = 1
        
        term = sigma(1-self.__fw(X))*(self.f(X)-y)/(1-self.f(X))
        return term * x_prime
    
    def predict(self, X):
        """
        Predict the target values for the given data.
        
        :param X: Data
        :return: Predicted target values
        """
        return np.where(self.f(X) >= 0.5, 1, 0)

    def fit(self, X, y, learning_rate=0.01, epochs=1000, verbose=False, random_state=42, mode = 'batch'):
        """
        Fit the linear classifier to the training data.
        
        :param X: Training data
        :param y: Target values
        :param learning_rate: Learning rate for gradient descent
        :param epochs: Number of iterations for gradient descent
        """
        self.ndim = X.shape[1]
        if mode=='batch': 
            return self.batch_gd(X, y, lr = learning_rate, epochs = epochs, verbose = verbose, random_state = random_state)
        elif mode=='stochastic':
            return self.stochastic_gd(X, y, lr = learning_rate, epochs = epochs, verbose = verbose, random_state = random_state)