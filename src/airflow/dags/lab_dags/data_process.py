import logging
import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import mlflow

logger = logging.getLogger(__name__)


TEST_SIZE = 0.3
RANDOM_STATE = 26
DATA_URI = "https://raw.githubusercontent.com/Illania/MLOPS3-Lab3/main/data-raw/data.csv"


def _to_numpy(df: pd.DataFrame) -> np.ndarray:
    """This private function takes a pandas DataFrame df as input and converts it into a NumPy array. 
    It first extracts the values from the DataFrame using the values attribute and stores it in a 
    variable y. Then, it expands the dimensions of y by adding a new axis at index 1 using the 
    expand_dims function from NumPy. Finally, it returns the modified y as a NumPy array.
    
    Parameters:
    ----------
    df: pd.DataFrame
        A pandas DataFrame containing the data to be converted.

    Returns:
    -------
    np.ndarray
        A NumPy array representing the converted DataFrame.
    """

    y = df.values
    y = np.expand_dims(y, axis=1)
    return y


def _separate_target_data(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """This private function takes a pandas DataFrame df and a column name column_name as input. 
    It separates the target variable from the DataFrame by storing it in a separate variable y. 
    It then removes the target variable column from the DataFrame using the drop function and 
    returns the modified DataFrame and the target variable y.

    Parameters:
    ----------
    df: pd.DataFrame
        A pandas DataFrame containing the data.
    
    column_name: str
        A string representing the name of the target variable column.

    Returns: 
    -------
    pd.DataFrame
        A pandas DataFrame representing the modified DataFrame with the target variable column 
        removed, and a pandas DataFrame y containing the target variable data.
    """

    y = df[column_name]
    df.drop(columns=[column_name], inplace=True)
    return y


def _split_features(df: pd.DataFrame) -> tuple:
    """This private function takes a pandas DataFrame df as input and splits the features 
    into two separate lists: num_columns for numeric columns and cat_columns for categorical columns. 
    It then returns a tuple containing these two lists.
    
    Parameters:
    ----------
    df: pd.DataFrame
        A pandas DataFrame containing the data to be analyzed.

    Returns: 
    -------
    tuple
        A tuple consisting of two list.
    """

    column_names = df.columns.to_list()
    cat_columns = []
    num_columns = []

    for column_name in column_names:
        if (df[column_name].dtypes == "int64") or (df[column_name].dtypes == "float64"):
            num_columns += [column_name]
        else:
            cat_columns += [column_name]
    return num_columns, cat_columns


def read_data(**kwargs) -> tuple:
    """This function reads data from a file specified by the DATA_URI variable, 
    splits it into training and testing sets using the train_test_split function, 
    and returns the training and testing sets as data_train and data_test, respectively.

    Parameters:
    ----------
    kwargs
        A variable-length keyword argument used in the function.

    Returns:
    -------
    tuple 
        A tuple containing the training and testing data sets, data_train and data_test, respectively.
    """

    df = pd.read_csv(DATA_URI)
    data_train, data_test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    logger.info("Data reading successfully done")
    return data_train, data_test


def preprocess_data(**kwargs) -> tuple:
    """This function preprocesses the input data for a machine learning model. It takes keyword arguments 
    kwargs as input. It uses the input argument "ti" from kwargs to fetch training and testing data using 
    the xcom_pull method. It then separates the target variable column "price" from the training and testing 
    data using the _separate_target_data function and converts them into NumPy arrays using the _to_numpy function.
    Next, it splits the features of the training data into numerical and categorical columns using the _split_features 
    function. It then applies preprocessing transformations such as scaling the numerical columns using StandardScaler 
    and encoding the categorical columns using OneHotEncoder with specific parameters. These transformations are organized 
    using make_column_transformer.
    Finally, it applies the preprocessors to both the training and testing data and returns the preprocessed training and 
    testing data along with their respective target variables.

    Parameters:
    ----------
    kwargs
        Keyword arguments passed to the function.

    Returns:
    -------
    tuple
        A tuple containing the preprocessed training data, training target variable, preprocessed testing data, and testing 
        target variable.
    """
    ti = kwargs["ti"]
    x_train, x_test = ti.xcom_pull(task_ids="read_data")

    y_train = _to_numpy(_separate_target_data(x_train, "price"))
    y_test = _to_numpy(_separate_target_data(x_test, "price"))

    num_cols, cat_cols = _split_features(x_train)

    preprocessors = make_column_transformer(
        (StandardScaler(), num_cols), (OneHotEncoder(drop="if_binary", handle_unknown="ignore"), cat_cols)
    )

    x_train = preprocessors.fit_transform(x_train)
    x_test = preprocessors.transform(x_test)

    return x_train, y_train, x_test, y_test


def prepare_model(**kwargs):
    """
    Trains a linear regression model on preprocessed data and logs the model and its parameters using MLflow.

    Parameters:
    ----------
    kwargs
        Keyword arguments passed to the function.

    Returns:
    -------
    str
        URI of the logged model.
    """

    ti = kwargs["ti"]
    x_train, y_train, _, _ = ti.xcom_pull(task_ids="preprocess_data")

    params = dict(
        fit_intercept=True,
        n_jobs=-1,
    )
    model = LinearRegression(**params)
    model.fit(x_train, y_train)

    mlflow.set_tracking_uri("http://mlflow_server:5000")
    try:
        mlflow.create_experiment("demo_data_process_flow")
    except Exception as e:
        logging.info(f"Got exception when mlflow.create_experiment: {e}")
    experiment = mlflow.set_experiment("demo_data_process_flow")
    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        mlflow.log_params(params)

        result = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="models",
            registered_model_name="LinearRegression-reg-model",
        )
        return result.model_uri


def check_model(**kwargs):
    """Evaluate a trained machine learning model using test data.
    Summary:
    - The function retrieves the test data (x_test and y_test) from XCom using the task ID "preprocess_data".
    - It also retrieves the model URI obtained from the "prepare_model" task and logs it as model_uri.
    - The MLflow tracking URI is set to "http://mlflow_server:5000" for logging the evaluation metrics.
    - The trained model is loaded using `mlflow.pyfunc.load_model` by providing the model URI.
    - Predictions are made on the test data (x_test) using the loaded model.
    - An MLflow experiment with the name "demo_data_process_flow_check_model" is set up.
    - Within the experiment run, the function logs the evaluation metrics: MSE, RMSE, and R2 using `mlflow.log_metrics`.

    Parameters:
    ----------
    kwargs
        Dictionary of keyword arguments, expected to contain the following:
      - ti: TaskInstance object (required) from Airflow, contains context information.
    
    Returns:
    -------
        None
    """

    ti = kwargs["ti"]
    _, _, x_test, y_test = ti.xcom_pull(task_ids="preprocess_data")
    model_uri = ti.xcom_pull(task_ids="prepare_model")
    logger.info(f"model_uri: {model_uri}")

    mlflow.set_tracking_uri("http://mlflow_server:5000")

    model = mlflow.pyfunc.load_model(model_uri=model_uri)

    y_pred = model.predict(x_test)

    experiment = mlflow.set_experiment("demo_data_process_flow_check_model")
    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        mlflow.log_metrics(
            {"MSE": mse(y_test, y_pred), "RMSE": mse(y_test, y_pred, squared=False), "R2": r2_score(y_test, y_pred)}
        )
