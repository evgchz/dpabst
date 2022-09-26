# dpabst
Post-processing algorithm for binary classification with abstention and DP constraints


# Warnings

1. Due to this bug https://github.com/scikit-optimize/scikit-optimize/issues/981 use `LogisticRegression` with `solver='liblinear'`

2. It is explicitly assumed that the sensitive attribute is stored in the LAST column of the design matrix.

3. It is explicitly assumed that the labels are valued in 0,1.

4. The prediction is 0, 1, or 10000. Reject is interpreted as 10000.

# Dependencies

This code was running with the following packages

scipy 1.6.0

numpy 1.20.0

fairlearn 0.5.0

sklearn 0.23.2


# Minimal example

```python
from sklearn.linear_model import LogisticRegression
from dpabst.post_process import TransformDPAbstantion
import numpy as np


# Last column contains group info
X_train = np.array([[0., 0., 0.], [0., 0., 1.]])
y_train = np.array([0., 1.])


X_unlab = np.array([[0., 0., 0.], [0., 0., 1.]])


X_test = np.array([[0., 0., 0.], [0., 0., 1.]])


clf = LogisticRegression(solver='liblinear')
clf.fit(X_train, y_train)


alphas = {0: .1, 1: .1}
transformer = TransformDPAbstantion(clf, alphas)
transformer.fit(X_unlab)
y_pred = transformer.predict(X_test)
print(y_pred)
```

# Bibentry
```bib
@article{Schreuder_Chzhen21,
      title={Classification with abstention but without disparities}, 
      author={N. Schreuder and E. Chzhen},
      year={2021},
      journal={UAI2021}
}

