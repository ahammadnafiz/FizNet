# ZyLearn 🚀

**Version:** 0.1.0

ZyLearn is an under-development Python library that provides a collection of machine-learning models for various tasks, including classification and beyond. Currently, it features implementations of logistic regression, neural networks, and k-nearest neighbors (KNN) classifiers. Designed for ease of use, ZyLearn allows researchers and practitioners to quickly experiment with different machine learning techniques.

Note: ZyLearn is still under active development and has not yet been packaged for distribution. Future releases will include more algorithms and features.

## Features ✨

- **Logistic Classifier:** A robust model for binary classification with an easy-to-use API.
- **Neural Network:** Flexible architecture for both simple and complex tasks.
- **KNN Classifier:** Intuitive classification model based on feature similarity.
- **Upcoming Additions:** New algorithms and tools are planned for future versions.

## Getting Started 🏁

To start using ZyLearn, import and train the desired model on your dataset. Here's a quick example showcasing how to use the available models:

```python
import numpy as np
from ZyLearn.logistic_regression import LogisticClassifier
from ZyLearn.neural_network import NN
from ZyLearn.knn_classifier import KNNClassifier

# Sample data
X_train = np.random.rand(100, 10)
y_train = np.random.randint(0, 2, size=100)
X_test = np.random.rand(20, 10)

# Initialize models
logistic_model = LogisticClassifier()
nn_model = NN(layer_dims=[X_train.shape[1], 10, 5, 2])
knn_model = KNNClassifier()

# Train models
logistic_model.train(X_train, y_train, num_iterations=2500, learning_rate=0.1)
nn_model.train(X_train, y_train, num_iterations=2500, learning_rate=0.1)
knn_model.train(X_train, y_train)

# Make predictions
logistic_predictions = logistic_model.predict(X_test)
nn_predictions = nn_model.predict(X_test)
knn_predictions = knn_model.predict(X_test)
```

## Available Models 🛠️

ZyLearn currently provides the following models for classification tasks:

- **LogisticClassifier**
- **NN (Neural Network)**
- **KNNClassifier**

You can list the available models programmatically:

```python
from ZyLearn import get_available_models

models = get_available_models()
print(models)
```

## Contributing 🤝

We invite contributions! Whether it's submitting issues, forking the repository, or creating pull requests, we'd love your help. Please open an issue to discuss your proposal before starting work for larger changes.

### Steps to Contribute

1. Fork the repository.
2. Create a new feature branch.
3. Implement your changes.
4. Ensure tests pass and update documentation as needed.
5. Submit a pull request.

## License 📝

This project is licensed under the MIT License. See the [LICENSE](https://github.com/ahammadnafiz/ZyLearn/blob/main/LICENSE) file for details.
