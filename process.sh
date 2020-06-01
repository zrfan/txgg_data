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

tf_jar_path="--jars zhongrui3/depend/spark-tensorflow-connector_2.11-1.10.0.jar"
$SPARK_HOME/bin/spark-submit \
    --files  $tf_jar_path \
    --name "txgg-data-preprocess" \
    --class txgg.data.prcoess.FeatureProcess \
    ./target/txgg_data-1.0-SNAPSHOT.jar $funcname