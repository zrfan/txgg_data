package txgg.data.process

import org.apache.spark.sql.{SQLContext, SparkSession}

object FeatureProcess {
	def main(args: Array[String]): Unit = {
		if (args.length == 0) print("Please set parameter")
		println("args=", args.mkString(", "))
		//        val params = SparkUtil.parse(args)
		//        val dt = params.getOrElse("dt", "20200227")
		val dt = args(0)
		val predt = args(1)
		val target_path = args(2)
		
		val sparkSession = SparkSession.builder().appName("TxggDataTest")
			.getOrCreate()
		val sparkContext = sparkSession.sparkContext
		val sparkConf = sparkContext.getConf
		val sqlContext = new SQLContext(sparkContext)
		val numPartitions = 460
		val dataPath = "hdfs://ns3-backup/user/weibo_bigdata_push/zhongrui3/txgg/"
		println("dataPath=", dataPath)
	}

	
}
