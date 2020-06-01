package txgg.data.process

import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{Row, SQLContext, SparkSession}
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
		val savePath = "/home/fzr/txgg/data/processed/"
		println("dataPath=", dataPath)
		println("funcname=", func_name)
		val train_ad_data = sparkSession.sparkContext.textFile(dataPath + "/train_preliminary/ad.csv")
			.map(p => p.split(","))
			.map(p => (p(0), p(1), p(2), p(3), p(4), p(5))).repartition(numPartitions)
			.map(p => (p._1,
				if (p._2 == "\\N") "4000000" else p._2,
					if (p._3 == "\\N") "60000" else p._3,
					if (p._4 == "\\N") "30" else p._4,
					if (p._5 == "\\N") "63000" else p._5,
					if (p._6 == "\\N") "400" else p._6))
			.persist(StorageLevel.MEMORY_AND_DISK)
		train_ad_data.take(10).foreach(println)
		
//		val train_click_data = sparkSession.read.format("csv").option("header", "true")
//			.load(dataPath + "/train_preliminary/click_log.csv").repartition(numPartitions).rdd
//			.map(p => (p.getAs[String]("creative_id"),
//				(p.getAs[String]("user_id"), p.getAs[String]("time"), p.getAs[String]("click_times"))))
//			.repartition(numPartitions).persist(StorageLevel.MEMORY_AND_DISK)
		val schema = StructType(List(
			StructField("creative_id", StringType), StructField("ad_id", StringType), StructField("product_id", StringType),
			StructField("product_category", StringType), StructField("advertiser_id", StringType), StructField("industry", StringType)
		))
		val predict_ad_data = train_ad_data
			.map(p => Row(p._1, p._2, p._3, p._4, p._5, p._6)) // user_id&label, ad_seq
		val predict_ad_df = sparkSession.createDataFrame(predict_ad_data, schema)
		println("predict result")
		predict_ad_df.show(20, false)
		println("predict count=", predict_ad_df.count())
		predict_ad_df.repartition(2).write.format("tfrecords").option("recordType", "Example")
			.mode("overwrite").save(savePath + s"/txpredict.tfrecords")
		
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
