#!/bin/sh
# sh material.sh 20200204  >> push.log 2>&1 &

base_func="new_user_list"  #

echo "get params: $*"

#function name
if [ $# -gt 0 ]
then
    funcname=$1
else
    funcname=$base_func
fi

echo "funcname=$funcname"

mvn package

tf_jar_path="--jars /home/fzr/env/spark-tensorflow-connector_2.11-1.10.0.jar"
#
spark-submit --name "txgg-data-preprocess" \
    $tf_jar_path \
    --class txgg.data.process.FeatureProcess \
    ./target/txgg_data-1.0-SNAPSHOT.jar $funcname  > spark.log 2>&1 &

tail -f spark.log