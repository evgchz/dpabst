# dpabst
Post-processing algorithm for binary classification with abstention and DP constraints


# Warning

Due to the fact that the linear program i solved with scipy, sometimes we run into the this bug: https://github.com/scikit-optimize/scikit-optimize/issues/981

The reason for this bug is not clear to us, if we face it, we run the code again and it disappears ... (WTF?)

# Dependencies

This code was running with the following packages

scipy 1.6.0
numpy 1.20.0
fairlearn 0.5.0
sklearn 0.23.2