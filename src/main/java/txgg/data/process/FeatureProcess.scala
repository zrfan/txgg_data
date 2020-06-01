package txgg.data.process

import org.apache.spark.sql.functions.{count, lit, sum}
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{Dataset, Row, SQLContext, SparkSession}
import org.apache.spark.storage.StorageLevel

object FeatureProcess {
	def main(args: Array[String]): Unit = {
		if (args.length == 0) print("Please set parameter")
		println("args=", args.mkString(", "))
		val func_name = args(0)
		
		val sparkSession = SparkSession.builder().appName("TxggDataTest") // 2.2.1
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
			.map(p => (p(0), p(1), p(2), p(3), p(4), p(5)))
		
		val test_ad_data = sparkSession.sparkContext.textFile(dataPath + "/test/ad.csv")
			.map(p => p.split(","))
			.map(p => (p(0), p(1), p(2), p(3), p(4), p(5)))
		val all_ad_data = train_ad_data.union(test_ad_data).repartition(numPartitions)
			.map(p => (p._1,
				if (p._2 == "\\N") "4000000" else p._2,
				if (p._3 == "\\N") "60000" else p._3,
				if (p._4 == "\\N") "30" else p._4,
				if (p._5 == "\\N") "63000" else p._5,
				if (p._6 == "\\N") "400" else p._6)).distinct()
			.persist(StorageLevel.MEMORY_AND_DISK)
		all_ad_data.take(10).foreach(println)  // 去重后广告数3412773
		println("all_ad_data count=", all_ad_data.count())
		val train_click_data = sparkSession.read.format("csv").option("header", "true")
			.load(dataPath + "/train_preliminary/click_log.csv")
		val test_click_data = sparkSession.read.format("csv").option("header", "true")
			.load(dataPath + "/test/click_log.csv").repartition(numPartitions)
		val all_click_data = train_click_data.union(test_click_data).repartition(numPartitions)
			.persist(StorageLevel.MEMORY_AND_DISK)
		println("all_click_data count=", all_ad_data.count())
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
		
		
		if (func_name == "new_user_list") {
//			userFeatureProcess(sparkSession, dataPath, numPartitions)
		} else if(func_name == "adfeature"){
//			adFeatureProcess(sparkSession, dataPath, numPartitions)
		}
		else {
			println("please enter a funcname: userfeature / sequence_uid_ad / makegraph / ad_list")
		}
		
	}
	def userFeatureProcess(user_data: Dataset[Row], sparkSession: SparkSession, dataPath: String, numPartitions: Int): Unit={
	
	}
	def adFeatureProcess(ad_data: Dataset[Row], sparkSession: SparkSession, dataPath: String, numPartitions: Int): Unit={
		
		println("origin_data describe")
		ad_data.describe().show(false)
		//		println("describe")
		// 全部用户的平均点击数：35.679
		
		val K = 2
		val splits = ad_data.randomSplit(Array(0.5, 0.5), seed = 2020L)
		//		, "product_category", "advertiser_id"
		val feature_names = Array("industry")
		//		, "gender"
		val target_features = Array("age")
		val target_val = Array("click_times")
		
		for (k <- Array.range(0, K)){
			splits(k) = splits(k).withColumn("fold", lit(k))
			println("split=", k)
			//			splits(k).show(false)
			if (k>0){
				splits(0) = splits(0).union(splits(k))
			}
		}
		val fold_data = splits(0).persist(StorageLevel.MEMORY_AND_DISK)
		ad_data.unpersist()
		var encoded_data: Dataset[Row] = null
		// 五折统计目标编码
		for (k <- Array.range(0, K)){
			var k_fold = fold_data.filter("fold="+k)
			println("start process ", k," fold data")
			k_fold.show(false)
			val other = fold_data.filter("fold!="+k)
			val encoder = targetEncoder(other, k_fold, feature_names, target_features, target_val, k)
			println("after targetEncoder, res=")
			encoder.show(20, false)
			if (k>0){
				encoded_data.union(encoder)
			}else encoded_data = encoder
		}
		println("encoded_data count=", encoded_data.count())
		encoded_data.show(20, false)
		
	}
	
	// 目标编码
	def targetEncoder(other: Dataset[Row], k_fold: Dataset[Row], feature_names: Array[String],
	                  target_features: Array[String], target_vals: Array[String], fold_num:Int):Dataset[Row]={
		var res = k_fold
		// compute target mean
		// 构造特征：被各age点击的概率，各age的人均点击次数，各age的人均每天点击次数，各age人每天最多点击次数，各age人每天最少点击次数
		for (feature_name <- feature_names){
			for (target_feature <- target_features){
				for(target_val <- target_vals){
					val tmp = other.select(feature_name, target_feature, target_val, "user_id", "time").persist(StorageLevel.MEMORY_AND_DISK)
					println("feature_name=", feature_name, "    target=", target_feature, "    Target_val=", target_val, "   count=", tmp.count())
					tmp.show(false)
					// feature_name 的点击总和
					val emperical_df = tmp.groupBy(feature_name).agg(sum(target_val).as(feature_name+"_"+target_val+"_sum"),
						count("user_id").as(feature_name+"_"+"user_id"+"_count"))
						.orderBy(feature_name)
					println("emperical grouped by feature_name=", feature_name)
					emperical_df.show(false)
					
					val grouped_target_df = tmp.groupBy(feature_name, target_feature)
					// 各age的点击次数总和，  求被各age点击的概率使用
					val clicks_sum = grouped_target_df.agg(sum(target_val).as(feature_name+"_"+target_feature+"_"+target_val+"_sum"),
						count("user_id").as(feature_name+"_"+target_feature+"_"+"user_id"+"_count"))
						.orderBy(feature_name, target_feature)
					println("grouped by", feature_name, " and ", target_feature, " clicks sum=")
					clicks_sum.show(false)
					//
					var click_likelihood = clicks_sum.join(emperical_df, usingColumn = feature_name)
					click_likelihood = click_likelihood.withColumn(target_feature+"_click_likelihood",
						click_likelihood(feature_name+"_"+target_feature+"_"+target_val+"_sum")/click_likelihood(feature_name+"_"+target_val+"_sum"))
						.withColumn("user_avg_click",
							click_likelihood(feature_name+"_"+target_feature+"_"+target_val+"_sum")/click_likelihood(feature_name+"_"+target_feature+"_"+"user_id"+"_count"))
					println("click likelihood join res=")
					click_likelihood.show(20, false)
					
					// transform result
					res = k_fold.join(click_likelihood, usingColumns = Seq(feature_name, target_feature))
					println("target encoder join to res by [", feature_name, target_feature, "]")
					res.show(20, false)
					
					// 各age的点击用户数，及人均点击次数
					//					val user_count_avg = grouped_target_df.agg((target_val, "sum"), ("user_id", "count"))
					//					println("clicks avg=")
					//					user_count_avg.show(false)
					//					// 各age的人每天点击次数
					//					val grouped_person_df = tmp.groupBy(feature_name, target, "user_id")
					//					val avg_click = grouped_person_df.agg((target_val, "sum"))
					//					println("person clicks avg=")
					//					avg_click.show(20, false)
					tmp.unpersist()
				}
			}
		}
		res
	}
	

	
}
