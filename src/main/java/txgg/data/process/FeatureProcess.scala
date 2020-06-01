package txgg.data.process

import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.spark.storage.StorageLevel

object FeatureProcess {
	def main(args: Array[String]): Unit = {
		if (args.length == 0) print("Please set parameter")
		println("args=", args.mkString(", "))
		val func_name = args(0)
		
		val sparkSession = SparkSession.builder().appName("TxggDataTest")
			.getOrCreate()
		val sparkContext = sparkSession.sparkContext
		val sparkConf = sparkContext.getConf
		val sqlContext = new SQLContext(sparkContext)
		val numPartitions = 460
		val dataPath = "/home/fzr/txgg/data/origin/"
		println("dataPath=", dataPath)
		println("funcname=", func_name)
		val train_path = dataPath+"/train_preliminary/"
		val test_path = dataPath+"/test/"
		val train_ad_data = sparkSession.sparkContext.textFile(train_path + "/ad.csv")
			.map(p => p.split(","))
			.map(p => (p(0), p(1), p(2), p(3), p(4), p(5))).repartition(numPartitions)
			.persist(StorageLevel.MEMORY_AND_DISK)
		train_ad_data.take(10).foreach(println)
//		val test_ad_data = sparkSession.sparkContext.textFile(test_path+"/ad.csv")
//			.map(p => p.split(","))
//			.map(p => (p(0), p(1), p(2), p(3), p(4), p(5))).repartition(numPartitions)
//			.persist(StorageLevel.MEMORY_AND_DISK)
		if (func_name == "userfeature") {
//			userFeatureProcess(sparkSession, dataPath, numPartitions)
		} else if(func_name == "adfeature"){
//			adFeatureProcess(sparkSession, dataPath, numPartitions)
		}
		else {
			println("please enter a funcname: userfeature / sequence_uid_ad / makegraph / ad_list")
		}
		
	}
	

	
}
