import numpy as np 
from tqdm import trange

def mse(y_true: np.ndarray, y_pred: np.array) -> float:
    """
    Calculates the MSE loss for the given `y_true` and `y_pred`.
    n is the number of samples.

    Parameters
    ----------
    y_true : np.ndarray
        The ground truth, should be of shape (n, 1)
    y_pred : np.array
        The predictions, should be of shape (n, 1)

    Returns
    -------
    float
        The computed MSE loss. 
    """
    assert y_true.shape == y_pred.shape, "`y_true` must have the same shape as `y_pred`."
    return np.sum(np.square(y_true - y_pred)) / y_true.shape[0]


def train_val_split(x: np.ndarray, y: np.ndarray, valid_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
    """
    Splits the data `x` and `y` into train and validation splits. 
    n is the number of samples and k is the number of features. 

    Parameters
    ----------
    x : np.ndarray
        Data `x` of shape (n, k)
    y : np.ndarray
        Data `y` of shape (n, 1)
    valid_ratio : float
        Fraction of the data used for validation. 

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        x_train, x_val, y_train, y_val, respectively.
    """
    n = x.shape[0]
    num_train = n - int(valid_ratio * n)
    x_train, x_val = x[:num_train, :], x[num_train:, :]
    y_train, y_val = y[:num_train, :], y[num_train:, :]
    return x_train, x_val, y_train, y_val

def linear_regression(x: np.ndarray, y: np.ndarray, epochs: int = 100, lr: float = 0.01, valid_ratio: float = 0.1, loss_func: callable = mse) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Fits a linear regression function to the data given with stochastic gradient descent. 

    Parameters
    ----------
    x : np.ndarray
        X data, should be of shape (n, k)
    y : np.ndarray
        y data, should be of shape (n, 1)
    epochs : int, optional
        Number of epochs, by default 100
    lr : float, optional
        Learning rate for gradient descent, by default 0.001
    valid_ratio : float, optional
        Fraction of data used in the validation, by default 0.1
    loss_func : callable, optional
        Loss function, by default mean squared error

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        Weight, bias, loss
    """

    # randomly initialize W, b according to the shape of x and y
    n, k = x.shape[0], x.shape[1]
    w = np.random.randn(k, 1)
    b = np.random.randn(1)

    # getting train and val splits for both x and y
    x_train, x_val, y_train, y_val = train_val_split(x, y, valid_ratio)

    with trange(epochs, desc="Training Epochs") as pbar:
        
        for epoch in pbar:
            
            total_loss = 0
            for i in range(len(x_train)):
                # computing loss
                y_pred = np.dot(w.T, x_train[i]) + b
                loss = loss_func(y_train[i], y_pred)
                total_loss += loss
                # calculating the gradient
                e = y_train[i] - x_train[i] @ w - b
                db = (-2/n) * np.sum(e)
                dw = (-2/n) * np.sum(x_train[i] * e)
                # update parameters
                w = w - lr * dw
                b = b - lr * db
            
            # calculate average loss for the epoch
            avg_loss = total_loss / n
            pbar.set_postfix_str(f"Epoch: {epoch + 1}, Avg_loss: {avg_loss:.4f}")

    
    # validation
    y_pred = x_val @ w + b
    loss = loss_func(y_val, y_pred)

    return (w, b, loss)

def main():
    """Runs a test run of the linear regression"""
    
    # setting up test values for a sanity check
    np.random.seed(0)
    n = 2000
    k = 30
    X = np.random.rand(n, k)
    W = np.random.randn(k, 1)
    b = np.random.randn(1)
    Y = X @ W + b
    
    weight, bias, loss = linear_regression(X, Y, epochs=100, lr=2)
    print(weight, bias, loss)
    
if __name__ == "__main__":
    main()