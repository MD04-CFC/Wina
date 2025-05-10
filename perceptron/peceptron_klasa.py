import numpy as np


class Perceptron:
    def __init__(self, eta=0.004, epochs=1000,is_verbose = False):
        self.eta = eta
        self.epochs = epochs
        self.is_verbose = is_verbose
        self.list_of_errors = []
        
        
    def predict(self, x):
        total_stimulation = np.dot(x, self.w)       
        total_stimulation = np.dot(x, self.w)
        return np.where(total_stimulation > 200, 1, 0)
        
    
    def fit(self, X, y):
        self.list_of_errors = []
        self.w = np.random.rand(X.shape[1])
        for e in range(self.epochs):
            number_of_errors = 0 
            for x, y_target in zip(X,y):
                y_pred = self.predict(x)
                delta_w = self.eta * (y_target - y_pred) * x
                self.w += delta_w
                number_of_errors += 1 if y_target != y_pred else 0
            self.list_of_errors.append(number_of_errors)
        
            if(self.is_verbose):
                print("Epoch: {}, weights: {}, number of errors {}".format(
                        e, self.w, number_of_errors))
        

    def score(self, X, y_true):
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y_true)
        return accuracy
