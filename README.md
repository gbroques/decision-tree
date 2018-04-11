# Decision Tree Classifier

[![Build Status](https://travis-ci.org/gbroques/decision-tree.svg?branch=master)](https://travis-ci.org/gbroques/decision-tree)
[![Coverage Status](https://coveralls.io/repos/github/gbroques/decision-tree/badge.svg?branch=master)](https://coveralls.io/github/gbroques/decision-tree?branch=master)

## Usage
For the following dataset:

| feature 0 | feature 1 | feature 2 | label |
|-----------|-----------|-----------|-------|
| 1         | 1         | 1         | 0     |
| 1         | 1         | 0         | 1     |
| 0         | 0         | 1         | 1     |
| 1         | 1         | 0         | 0     |
| 1         | 0         | 0         | 1     |

```python
training_data = [
    [1, 1, 1, 0],
    [1, 1, 0, 1],
    [0, 0, 1, 1],
    [1, 1, 0, 0],
    [1, 0, 0, 1]
]
design_matrix = [row[:-1] for row in training_data]
target_values = [row[-1] for row in training_data]
decision_tree = DecisionTree()
decision_tree.fit(design_matrix, target_values)
predictions = decision_tree.predict(design_matrix)
```

To print the tree:
```python
decision_tree.print()
```

Which would output the following:
```
Is feature 1 >= 1?
--> True:
  Is feature 2 >= 1?
  --> True:
    Predict {0: '100.0%'}
  --> False:
    Predict {1: '50.0%', 0: '50.0%'}
--> False:
  Predict {1: '100.0%'}
```

In the case of ambiguous records like `[1, 1, 0]` where two records exist with the same feature values,
but different labels, the tree always predicts the first key in the prediction dictionary or `1` in this example.

Adapted from:

[Let's Write a Decision Tree Classifier from Scratch - Machine Learning Recipes #8](https://www.youtube.com/watch?v=LDRbO9a6XPU)
