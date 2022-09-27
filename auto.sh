#!/bin/sh
cd home/bdlt/WARC-DL2/satf/
echo "$1"
PYTHONPATH=. HADOOP_CONF_DIR=./hadoop/ HADOOP_USER_NAME=$USER python3 examples/future_extraction/future_extraction_pipeline.py $1;
exit
