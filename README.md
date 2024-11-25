### 11/21/24: Logistic Regression from scratch

### 11/24/24
- Finished v1 implementation
- Finished v1 model example. Running fit with hyperparameters, learning rate = 0.005, and 1000 epochs seems to not be training enough. Cross entropy error continues to go down

- with learning rate = 0.01, and 3000 epochs, cross entropy loss goes down to 0.60, which is still not that good.

## fixes
- I forgot to add the bias in my forward function. This messed up my predictions but also my update function
- in my update function, my weights gradient variable was a scalar instead of an array, despite the fact that my weights were an array

- with learning rate= 0.001, and 5000 epochs, we get down to cross entropy loss of 0.41, and accuracy of about 0.8. 

- with learning rate=0.0005 and 10,000 epochs, we get down to cross entropy loss of 0.37315917015075684, with accuracy 0.892. 

- last experiment, 30,000 epochs and learning rate of 0.0005 gives us cross entropy loss of 0.22859561443328857,  with accuracy 0.944. 


### conclusions
- implemeting this was pretty easy, as it was pretty similar to linear regression, with just the forward functions being different 
- the training set was 2,000 examples, which is still a pretty small dataset but we still needed many more passes than Linear Regression to predict decently for some reason. This might be because the dataset only has 5 features, and maybe each feature wasn't as highly correlated as the features of the Housing dataset in Linear Regression. 





