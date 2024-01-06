# Building the DAG using the functions from data_process and model module
import datetime as dt

import pendulum
from lab_dags.data_process import *

# Project
from airflow import DAG
from airflow.operators.python import PythonOperator

# Declare Default arguments for the DAG
default_args = dict(
    start_date=pendulum.datetime(2024, 1, 6, tz="UTC"),
    catchup=False,
    is_paused_upon_creation=True,
    provide_context=True,
)

# creating a new dag
dag = DAG(
    "dataflow_process_dag",
    default_args=default_args,
    schedule_interval="0 3 * * *",
    max_active_runs=1,
)

read_data = PythonOperator(task_id="read_data", python_callable=read_data, dag=dag)

preprocess_data = PythonOperator(task_id="preprocess_data", python_callable=preprocess_data, dag=dag)

prepare_model = PythonOperator(task_id="prepare_model", python_callable=prepare_model, dag=dag)

check_model = PythonOperator(task_id="check_model", python_callable=check_model, dag=dag)

# Set the task sequence
read_data.set_downstream(preprocess_data)
preprocess_data.set_downstream(prepare_model)
prepare_model.set_downstream(check_model)
