class SupervisedModel:
    """
    Abstract class for supervised learning problems.

    Any supervised learning algorithm should inherit from this class.
    It provides a common interface for fitting and predicting.
    The class requires the following methods to be implemented:
    - `fit`: Fit the model to the training data.
    - `predict`: Predict the target values for the given data.
    - `score`: Compute the score of the model.
    """
    def __init__(self, name: str = 'Default name'):
        self.name = name
        self.model = None

    def fit(self, X, y):
        """
        Fit the model to the training data.
        :param X: Training data
        :param y: Target values
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def predict(self, X):
        """
        Predict the target values for the given data.
        :param X: Data to predict
        :return: Predicted target values
        """
        raise NotImplementedError("Subclasses should implement this method.")
    

class GradientDescentSupervisedModel(SupervisedModel):
    """
    Abstract class for supervised learning problems using gradient descent.

    This class provides methods for batch and stochastic gradient descent.
    It needs to implement specific supervised learning algorithms methods.

    For gradient descent, it requires the following methods to be implemented:
    - `grad`: Compute the gradient of the loss function.
    - `loss`: Compute the loss function.
    - `update_params`: Update the model parameters using the gradient.
    - `init_params`: Initialize the model parameters.
    """
    def __init__(self, name: str = 'Default name'):
        super().__init__(name = name)

    def grad(self, X, y):
        """
        Compute the gradient of the loss function.
        :param X: Data
        :param y: Target values
        :return: Gradient
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def loss(self, X, y):
        """
        Compute the loss function.
        :param X: Data
        :param y: Target values
        :return: Loss
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def update_params(self, delta, lr=0.01):
        """
        Update the model parameters using the gradient.
        :param delta: Gradient
        :param lr: Learning rate
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def init_params(self, random_state=None):
        """
        Initialize the model parameters.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def batch_gd(self, X , y , lr=0.01, epochs=100, verbose=False, random_state=42):
        """
        Perform batch gradient descent.
        :param X: Training data
        :param y: Target values
        :param lr: Learning rate
        :param epochs: Number of epochs
        :param verbose: If True, print the loss at each epoch

        :return: List of losses
        """
        losses = []
        self.init_params(random_state=random_state)
        n = X.shape[0]
        for i in range(epochs):
            delta = None
            for j in range(n):
                if delta is None: delta = self.grad(X[j], y[j])
                else: delta += self.grad(X[j], y[j])
            self.update_params(delta, lr=lr)
            losses.append(self.loss(X,y))
            if verbose: print( f"Epoch {i+1}/{epochs}, Loss: {losses[-1]}")
        return losses
    
    def stochastic_gd(self, X , y , lr=0.01, epochs=100, verbose=False, random_state=42):
        """
        Perform batch gradient descent.
        :param X: Training data
        :param y: Target values
        :param lr: Learning rate
        :param epochs: Number of epochs
        :param verbose: If True, print the loss at each epoch

        :return: List of losses
        """
        losses = []
        self.init_params(random_state=random_state)
        n = X.shape[0]
        for i in range(epochs):
            delta = None
            for j in range(n):
                delta = self.grad(X[j], y[j])
                self.update_params(delta, lr=lr)
            losses.append(self.loss(X,y))
            if verbose: print(f"Epoch {i+1}/{epochs}, Loss: {losses[-1]}")
        return losses