import numpy as np

class LogisticRegression():
    
    def __init__(self, num_epochs=10, learning_rate=0.01):
        '''
        init function, can pass number of epochs, learning rate, and proportion of the input data as test_size
        '''
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        # we also have a weights and bias, once again going to be a vector and a scalar 
        self.weights = None
        self.bias = None 

    def fit(self, train_data, train_labels):
        # define weights and bias 
        # np.random.uniform takes (low, high)
        self.bias = np.float32(np.random.uniform(-1, 1))
        # to get weights, we need number of features in the training_data
        num_training_examples, num_features = np.shape(train_data)
        # so the weights is gonna be (1 x m)
        self.weights = np.random.uniform(-1, 1, size=(1, num_features)).astype(np.float32)
        # then we'd run train? with the input_data??, after the split
        self.train(train_data, train_labels)

    def forward(self, x):
        exp = np.matmul(self.weights, x).astype(np.float32) + self.bias
        return self.sigmoid(exp).flatten()

    ## done 
    def predict(self, x):
        boundary = 0.5
        forward = self.forward(x)
        if forward >= boundary:
            return 1.0
        else:
            return 0.0
    
    ## done 
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def update(self, y, yhat, x):
        # thetaj := thetaj + alpha(sum(yi - h(theta)(xi))xji
        # h-theta(x) --> yhat this is the prediction 
        # sum gradients, and divide by len(y) to get the average gradient direction, and use it to update the weights vector. Make sure to multiply by input x.
        # bias, we use the same gradients calculation, but we're not going to use x, because bias is not related to x
        weights_grad = np.zeros_like(self.weights) 
        bias_grad = 0
        yhat = yhat.T
        for i in range(len(y)):
            yi = y[i]
            yhati = yhat[i]
            # need to 
            xi = x[i]

            weights_grad += (yi - yhati) * x[i]
            bias_grad += yi - yhati

        # update with learning rate after, make sure to average
        self.weights += self.learning_rate * weights_grad / len(y)
        self.bias += self.learning_rate * bias_grad / len(y)

    def cross_entropy_loss(self, y, yhat):
        yhat = np.clip(yhat, 1e-10, 1 - 1e-10)
        
        loss = -np.mean(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))
        
        return loss

    def train(self, train_inputs, train_labels):
        for epoch in range(self.num_epochs):
            print(f"Training epoch {epoch}")
            # get y, yhat for each x 
            yhat = self.forward(train_inputs.T)
            loss = self.cross_entropy_loss(train_labels, yhat)
            print(f"Cross Entropy Loss {loss}")
            self.update(train_labels, yhat, train_inputs)



''''
implementation thoughts:

- we're going to need a series of functions:
    - init, create an empty weights function, that fit will use
    - bias will once again be a random value 
    - weights, let's think about what the shape of it should be 
    - we are going to do thetaT * x
    - x is going to be (n x m), n is number of examples, m is the number of labels 
    - theta is therefore going to be (1 x m), so that thetaT = (m x 1)
    - giving us (n x m) x (m x 1), n x 1, outputs, which we can put in the sigmoid function
    - we are going to have the fit function do the training test split, we can include the split as an optional parameter in init 
'''
