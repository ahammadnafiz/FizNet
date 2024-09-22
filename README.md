# FizNet 🚀

**Version:** 0.1.0

FizNet is an under-development Python library that offers a collection of machine learning models for various tasks, including classification and beyond. Currently, it features implementations of logistic regression, neural networks, and k-nearest neighbors (KNN) classifiers. Designed for ease of use, FizNet enables researchers and practitioners to quickly experiment with different machine learning techniques.

Note: Currently, FizNet is not packaged for distribution and is under active development. More algorithms and features will be added in future releases.

## Table of Contents 📚

- [Features](#features)
- [Getting Started](#getting-started)
- [Available Models](#available-models)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## Features ✨

- **Logistic Classifier:** Efficient for binary classification tasks with a simple API.
- **Neural Network:** A flexible architecture for handling both simple and complex problems.
- **KNN Classifier:** An intuitive model for classification based on feature similarity.
- **Future Enhancements:** Additional algorithms and features are planned for future releases.
- **Future Enhancements:** Additional algorithms and features are planned for future releases.

## Getting Started 🏁

To get started with FizNet, import the desired model and fit it to your dataset. Here’s a quick example demonstrating how to use each classifier:

```python
import numpy as np
from FizNet.logistic_regression import LogisticClassifier
from FizNet.neural_network import NN
from FizNet.knn_classifier import KNNClassifier

# Sample data
X_train = np.random.rand(100, 10)
y_train = np.random.randint(0, 2, size=100)
X_test = np.random.rand(20, 10)

# Initialize the models
logistic_model = LogisticClassifier()
nn_model = NN(layer_dims=[X_train.shape[0], 10, 5, 2])
knn_model = KNNClassifier()

# Train the models
logistic_model.train(X_train, y_train, num_iterations=2500, learning_rate=0.1)
nn_model.train(X_train, y_train, num_iterations=2500, learning_rate=0.1 )
knn_model.train(X_train, y_train)

# Make predictions
logistic_predictions = logistic_model.predict(X_test)
nn_predictions = nn_model.predict(X_test)
knn_predictions = knn_model.predict(X_test)
```

## Available Models 🛠️

FizNet provides the following models for classification:

- **LogisticClassifier**
- **NN**
- **KNNClassifier**

You can programmatically retrieve the list of available models:

```python
from FizNet import get_available_models

models = get_available_models()
print(models)
```

## Contributing 🤝

We welcome contributions! Feel free to submit issues, fork the repository, and create pull requests. For larger changes, consider opening an issue to discuss your proposal before implementing it.

### Steps to Contribute

1. Fork the repository.
2. Create a new feature branch.
3. Implement your changes.
4. Ensure tests pass and update documentation if necessary.
5. Submit a pull request.

## License 📝

This project is licensed under the MIT License. See the [LICENSE](https://github.com/ahammadnafiz/FizNet/blob/main/LICENSE) file for details.