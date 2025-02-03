Linear Regression Implementation with PyTorch
A fundamental implementation of linear regression that demonstrates how to build, train, and evaluate a basic machine learning model using PyTorch.

What is Linear Regression?
Linear regression is a statistical method that attempts to model the relationship between variables by fitting a linear equation to observed data. In this project, we model:

y = ax + b

Where:

y is the predicted value (dependent variable)
x is the input value (independent variable)
a is the slope (weight)
b is the y-intercept (bias)
Implementation Details
Model Architecture
```python
class LinearRegressionM(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1))  # Slope
        self.b = nn.Parameter(torch.randn(1))  # Bias

    def forward(self, x):
        return self.a * x + self.b
```
Training Process
Generate synthetic data with known parameters (a=0.3, b=0.7)
Split data into training (80%) and test (20%) sets
Train model using:
Loss function: L1Loss (Mean Absolute Error)
Optimizer: SGD with learning rate 0.01
Epochs: 1000

# Create and train model 
model = LinearRegressionM()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.L1Loss()

for epoch in range(1000):
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train) 
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Save model
torch.save(model.state_dict(), 'models/linearRM.pth')

# Load model
loaded_model = LinearRegressionM()
loaded_model.load_state_dict(torch.load('models/linearRM.pth'))

Model Performance
The model successfully learns the underlying parameters:

Initial random values: a ≈ 0.34, b ≈ 0.13
Final learned values: a ≈ 0.31, b ≈ 0.70
Training loss: 0.003
Test loss: 0.012
Usage
Visualization
The project includes functions to visualize:

Training data points
Test data points
Model's predicted line
Loss curves during training
Requirements
Python 3.8+
PyTorch
Matplotlib (for visualization)
Key Learnings
This project demonstrates:

Building neural networks with PyTorch's nn.Module
Working with tensors and autograd
Implementing training loops
Model persistence (saving/loading)
Basic data visualization
Evaluating model performance
The final model successfully learns the linear relationship from noisy data, achieving loss values close to optimal.