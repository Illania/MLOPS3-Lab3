FROM apache/airflow:2.7.3-python3.10

COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /tmp/requirements.txt
# start M1 fixes
RUN pip uninstall psycopg2-binary -y
USER root
RUN apt-get update -y
RUN apt install -y build-essential libpq-dev
USER airflow
RUN pip install psycopg2
# end m1 fix
COPY --chown=airflow:root dags/lab_dags/ /opt/airflow/dags/lab_dags/

