import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from base_module import BaseModule
from base_learning_rate import  BaseLR
from gradient_descent import GradientDescent
from learning_rate import FixedLR
from sklearn.metrics import roc_curve, auc
from loss_functions import misclassification_error
from cross_validate import cross_validate



# from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from modules import L1, L2
from logistic_regression import LogisticRegression
from utils import split_train_test

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=dict(text=f"GD Descent Path {title}", font=dict(size=20))))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values_list, weights_list = [], []
    def callback(solver: GradientDescent, weights: np.ndarray, val: np.ndarray, grad: np.ndarray,
                 t: int, eta: float, delta: float) -> None:
        values_list.append(val)
        weights_list.append(weights.copy())
    return callback, values_list, weights_list

   


def plot_decsent_and_convergence(module: Type[BaseModule], 
                                 eta: float = 0.01,
                                 init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 xrange=(-1.5, 1.5),
                                 yrange=(-1.5, 1.5)) -> Tuple[go.Figure, go.Figure, float]:
    """
    This function is one iteration of the 'compere_fixed_learning_rates' loop.
    """
    callback, values, weights = get_gd_state_recorder_callback()
    # Initialize the fixed learning rate
    fixed_lr = FixedLR(base_lr=eta)
    # Initialize the gradient descent solver
    gd_solver = GradientDescent(learning_rate=fixed_lr, max_iter=1000, tol=1e-5, callback=callback)
    # Initialize the module
    module_instance = module(weights=init.copy())
    # Run gradient descent
    gd_solver.fit(module_instance, X=None, y=None)

    # Plot the descent path
    fig_descent = plot_descent_path(module, np.array(weights), title=f", {module.__name__} regularization, eta={eta}",
                                    xrange=xrange, yrange=yrange)
    # Plot the convergence
    trace_convergence = go.Scatter(y=[v[0] for v in values], mode='lines+markers', name=f"learning rate: {eta}")
    best_val = min(values, key=lambda v: v[0])
    return fig_descent, trace_convergence, best_val[0]


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    """
    This function evaluates and visualizes how gradient descent behaves with different fixed learning rates when
    Initializes the weights using the `init` array.
    Runs gradient descent using the FixedLR (fixed learning rate) scheduler.
    # Plot algorithm's descent path
    # Plot algorithm's convergence for the different values of eta
    """
    L1_descent_figs = []
    L2_descent_figs = []
    L1_conv_traces = []
    L2_conv_traces = []
    best_val_L1, best_val_L2 = float('inf'), float('inf')
    for eta in etas:
        fig_descent_L1, trace_convergence_L1, curr_best_val_L1 = plot_decsent_and_convergence(L1, eta=eta, init=init)
        fig_descent_L2, trace_convergence_L2, curr_best_val_L2 = plot_decsent_and_convergence(L2, eta=eta, init=init)
        L1_descent_figs.append(fig_descent_L1)
        L2_descent_figs.append(fig_descent_L2)
        L1_conv_traces.append(trace_convergence_L1)
        L2_conv_traces.append(trace_convergence_L2)
        if curr_best_val_L1 < best_val_L1:
            best_val_L1 = curr_best_val_L1
        if curr_best_val_L2 < best_val_L2:
            best_val_L2 = curr_best_val_L2
    # Show the descent path and convergence for each learning rate
    for fig in L1_descent_figs:
        fig.show()
    for fig in L2_descent_figs:
        fig.show()
    fig_L1_convergence = go.Figure(data=L1_conv_traces,
                               layout=go.Layout(title=dict(text="L1 Convergence for Different Learning Rates.\nBest value (approximately): {:.2e}".format(best_val_L1), font=dict(size=20)),
                                                xaxis_title="Iteration",
                                                yaxis_title="Objective Value"))

    fig_L2_convergence = go.Figure(data=L2_conv_traces,
                               layout=go.Layout(title=dict(text="L2 Convergence for Different Learning Rates.\nBest value (approximately): {:.2e}".format(best_val_L2), font=dict(size=20)),
                                                xaxis_title="Iteration",
                                                yaxis_title="Objective Value"))

    fig_L1_convergence.show()
    fig_L2_convergence.show()



def load_data(path: str = "SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)

def plot_convergence_logistic_regression(X_train: pd.DataFrame, y_train:pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
    include_intercept = True # TODO - this is ny descision, not asked in the task. 
    etas = (1, .1, .01, .001) # TODO - this is my descision, not asked in the task.
    regularizations = ["l1", "l2", "none"]
    log_convergence_traces = []
    for reg in regularizations:
        reg_trace = []
        for eta in etas:
            callback, values, weights = get_gd_state_recorder_callback()
            fixed_lr = FixedLR(base_lr=eta)
            gd_solver = GradientDescent(learning_rate=fixed_lr, max_iter=1000, tol=1e-6, callback=callback)
            log_reg = LogisticRegression(solver=gd_solver, include_intercept=include_intercept)
            
            log_reg.fit(X_train.to_numpy(), y_train.to_numpy())
            trace_convergence = go.Scatter(y=[v[0] for v in values], mode='lines+markers', name=f"learning rate: {eta}")
            reg_trace.append(trace_convergence)
        log_convergence_traces.append(reg_trace)

    fig_l1_convergence = go.Figure(data=log_convergence_traces[0],
                                layout=go.Layout(title=dict(text="LogisticRegression Convergence for Different Learning Rates (with L1 regularization, lambda=1, alpha=0.5)", font=dict(size=20)),
                                                    xaxis_title="Iteration",
                                                    yaxis_title="objective Value"))
    fig_l2_convergence = go.Figure(data=log_convergence_traces[1],
                                layout=go.Layout(title=dict(text="LogisticRegression Convergence for Different Learning Rates (with L2 regularization, lambda=1, alpha=0.5)", font=dict(size=20)),
                                                    xaxis_title="Iteration",
                                                    yaxis_title="objective Value"))
    fig_none_convergence = go.Figure(data=log_convergence_traces[2],
                                layout=go.Layout(title=dict(text="LogisticRegression Convergence for Different Learning Rates (without regularization, lambda=1, alpha=0.5)", font=dict(size=20)),
                                                    xaxis_title="Iteration",
                                                    yaxis_title="objective Value"))
    fig_l1_convergence.show()
    fig_l2_convergence.show()
    fig_none_convergence.show()
    

    

def plot_roc(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
    # Convert to NumPy arrays
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Fit model
    model = LogisticRegression() # default values
    model.fit(X_train, y_train)

    # Predict probabilities
    y_prob = model.predict_proba(X_test)

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    best_thr_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[best_thr_idx]
    test_error = np.mean((y_prob >= best_threshold).astype(int) != y_test)
    

    # Plot the ROC curve, and point the best threshold

    fig = go.Figure(
            data=[
                go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(color="black", dash='dash'), name="Random Class Assignment"),
                go.Scatter(
                    x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False, marker_size=5, marker_color="blue",
                    hovertemplate="<b>Threshold:</b>%{text:.6f}<br>FPR: %{x:.6f}<br>TPR: %{y:.6f}"
                ),
                go.Scatter(
                    x=[fpr[best_thr_idx]], y=[tpr[best_thr_idx]], mode='markers', marker=dict(color='red', size=12, symbol='star'),
                    name="Best Threshold", showlegend=True,
                    hovertemplate=f"<b>Best Threshold:</b> {best_threshold:.2e}<br>FPR: {fpr[best_thr_idx]:.6f}<br>TPR: {tpr[best_thr_idx]:.6f}"
                )
            ],
            layout=go.Layout(
                title=dict(text=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f},\ \text{{Test Error}}={test_error:.4f},\ \text{{Best Threshold}}={best_threshold:.5e}$", font=dict(size=20)),
                xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")
            )
    )
    fig.show()
    
    return fpr, tpr, thresholds, y_prob

def estimate_LogReg(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, penalty:str = 'none') -> None:

    X_train, y_train, X_test, y_test = X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()
    alpha, max_iter, lr = 0.5, 20000, 1e-4
    lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    results = []
    for lam in lambdas:
        gd_solver = GradientDescent(max_iter=max_iter, learning_rate=FixedLR(lr))
        model = LogisticRegression(solver=gd_solver, penalty=penalty, lam=lam, alpha=alpha)
        trains_score, val_score = cross_validate(model, X_train, y_train, misclassification_error)
        results.append((lam, trains_score, val_score))
    
    best_lambda, best_train, best_val = min(results, key=lambda t: t[2])  # minimize validation error

    # Fit model with best lambda and compute test error
    gd_solver = GradientDescent(max_iter=max_iter, learning_rate=FixedLR(lr))
    best_model = LogisticRegression(solver=gd_solver, penalty=penalty, lam=best_lambda, alpha=alpha)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    test_error = misclassification_error(y_test, y_pred)

    # plot line-graph of the train and validation errors for each lambda
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[r[0] for r in results], y=[r[1] for r in results], mode='lines+markers', name='Train Error'))
    fig.add_trace(go.Scatter(x=[r[0] for r in results], y=[r[2] for r in results], mode='lines+markers', name='Validation Error'))
    # Mark the best lambda
    fig.add_trace(go.Scatter(
        x=[best_lambda], y=[best_val], mode='markers', marker=dict(color='red', size=12, symbol='star'),
        name=f"Best Lambda ({best_lambda})"
    ))
    fig.update_layout(
        title=f"Train and Validation Errors for Different Lambda Values, with penalty={penalty.upper()}<br>Best Lambda={best_lambda}, Test Error={test_error:.4f}",
        xaxis_title='Lambda',
        yaxis_title='Error',
        legend=dict(title='Error Type')
    )
    fig.show()

    




def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data
    # --- plot_convergence_logistic_regression(X_train, y_train, X_test, y_test)--- TODO this part is depend on the answer from the forum


    plot_roc(X_train, y_train, X_test, y_test)
    


    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    estimate_LogReg(X_train, y_train, X_test, y_test, penalty='l1')
    # estimate_LogReg(X_train, y_train, X_test, y_test, penalty='l2') # TODO this part is depend on the answer from the forum

if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    fit_logistic_regression()