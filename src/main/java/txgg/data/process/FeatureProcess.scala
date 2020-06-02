package txgg.data.process

import com.microsoft.ml.spark.lightgbm.LightGBMClassifier
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql
import org.apache.spark.sql.functions.{count, lit, sum, udf, col}
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SQLContext, SparkSession}
import org.apache.spark.storage.StorageLevel

import scala.reflect.internal.util.TableDef.Column

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
		val numPartitions = 360
		val dataPath = "/home/fzr/txgg/data/origin/"
		val savePath = "/home/fzr/txgg/data/processed/"
		println("dataPath=", dataPath)
		println("funcname=", func_name)
		val all_ad_data = readAllAdData(sparkSession, dataPath, savePath, numPartitions).persist(StorageLevel.MEMORY_AND_DISK)
		println("all_ad data=")
		all_ad_data.show(false) // 去重后广告数3412773
		println("all_ad_data count=", all_ad_data.count())
		
		val all_click_data = readAllClickData(sparkSession, dataPath, savePath, numPartitions).persist(StorageLevel.MEMORY_AND_DISK) // user click age&gender
		println("all_click_data count=", all_click_data.count())
		
		println("all_ad_click_data_before_join")
		all_ad_data.show(50, false)
		all_click_data.show(50, false)
		
//		val full_click_data = all_click_data.join(all_ad_data, usingColumns = Seq("creative_id"), joinType = "left_outer")
//			.repartition(numPartitions)
//			.persist(StorageLevel.MEMORY_AND_DISK)
//		println("full click data after join")
//		full_click_data.show(50, false)
//
//
//		// 用户特征提取
//		val user_feature = userFeatureProcess(full_click_data, sparkSession, savePath, numPartitions)
//		println("user_feature")
//		user_feature.show(false)
//
//		val all_feature_cols = Array("all_click_cnt", "active_days", "creative_cnt", "ad_cnt", "product_cnt",
//			"category_cnt", "advertiser_cnt", "industry_cnt",
//			"mean_dur", "max_dur", "min_dur",
//			"max_click_product_id", "max_click_product_category", "max_click_advertiser_id", "max_click_industry")
//
//		val all_data = user_feature.select((all_feature_cols ++ Array("user_id", "age", "gender")).map(x => col(x)): _*)
//		println("all_data")
//		all_data.show(200, false)
//
//		val assembler = new VectorAssembler().setInputCols(all_feature_cols).setOutputCol("features")
//		val all_assembled_data = assembler.transform(all_data)
//		println("assembled")
//		all_assembled_data.show(false)
//		val all_train = all_assembled_data.filter("age!=0 and gender!=0").withColumn("label", user_feature("gender")*1.0-1.0)
//
//		// train
//		val lightgbm = new LightGBMClassifier().setLabelCol("label").setFeaturesCol("features")
//			.setPredictionCol("predict_label").setProbabilityCol("probability")
//		val Array(train, test) = all_train.randomSplit(Array(0.7, 0.3), seed = 2020L)
//		val model = lightgbm.fit(train)
//
//		val val_res = model.transform(test)
//		println("val_res=", val_res)
//		val_res.show(false)
//		val evaluator = new BinaryClassificationEvaluator().setLabelCol("label").setRawPredictionCol("predict_label")
////		val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("predict_label")
//		println("evalutor=", evaluator.evaluate(val_res))
		
		
		
		//predict
//		val predict = all_assembled_data.filter("age=0 and gender=0")
//		println("predict data=")
//		predict.show(false)
//		println("predict_data count=", predict.count())
//		val predict_res = model.transform(predict)
//		println("predict_res=")
//		predict_res.show(false)
		
		// 广告特征提取: 目标编码
//		val ad_train_feature = adTrainFeatureProcess(full_click_data.filter("age !=0 and gender != 0"),
//			sparkSession, dataPath, numPartitions)
		
		// 保存TFRecords文件
	}
	
	def userFeatureProcess(full_click_data: Dataset[Row], sparkSession: SparkSession, savePath: String, numPartitions: Int): Dataset[Row] = {
		val user_grouped = full_click_data.groupBy("user_id", "age", "gender")
		// 全部用户的平均点击数：35.679
		full_click_data.createTempView("txgg_temp")
		val user_agg_sql =
			s"""select user_id, age, gender, sum(click_times) as all_click_cnt, count(distinct time) as active_days,
			   | count(distinct creative_id) as creative_cnt,
			   | count(distinct ad_id) as ad_cnt, count(distinct product_id) as product_cnt,
			   | count(distinct product_category) as category_cnt, count(distinct advertiser_id) as advertiser_cnt,
			   | count(distinct industry) as industry_cnt,
			   | collect_list(time) as time_list, collect_list(creative_id) as creative_list, collect_list(ad_id) as ad_list,
			   | collect_set(product_id) as product_set, collect_set(product_category) as category_set,
			   | collect_set(advertiser_id) as advertiser_set, collect_set(industry) as industry_set
			   | from (select * from txgg_temp order by time) as A group by user_id, age, gender """.stripMargin
		var user_agg = sparkSession.sql(user_agg_sql)
		println("user_agg info")
		// 点击间隔统计
		def getDuring(time_list: scala.collection.mutable.WrappedArray[Int]): Array[Float] ={
			val dur_list = time_list.map(x => x.toFloat).sorted.sliding(2).map(x => x.last - x.head).toList
			val mean_dur = dur_list.sum/dur_list.length
			val max_dur = dur_list.max
			val min_dur = dur_list.min
			Array(mean_dur, max_dur, min_dur)
		}
		val durUDF = udf((time_list:scala.collection.mutable.WrappedArray[Int]) => {getDuring(time_list)})
		val user_dur = user_agg.withColumn("active_avg_clicks", user_agg("all_click_cnt")*1.0/user_agg("active_days"))
			.withColumn("dur", durUDF(col("time_list")))
			.select(col("user_id"), col("dur").getItem(0).as("mean_dur"),
				col("dur").getItem(1).as("max_dur"),
				col("dur").getItem(2).as("min_dur"))
		println("test mean dur")
		user_dur.show(false)
		user_agg = user_agg.join(user_dur, usingColumn = "user_id")
		// 最大点击特征统计
		val max_feature_names = Array("product_id", "product_category", "advertiser_id", "industry")
		for (name <- max_feature_names){
			val user_max_click_sql =
				s"""select b.user_id, b.$name as max_click_$name, b.cnt from (
				   |    select user_id, $name, cnt, row_number() over (partition by user_id order by cnt desc) rank
				   |    from ( select user_id, $name, sum(click_times) as cnt from txgg_temp group by user_id, $name) a
				   |  ) b where b.rank=1""".stripMargin
			val user_max_product = sparkSession.sql(user_max_click_sql)
			println("user_max_click data")
			user_max_product.show(false)
			user_agg = user_agg.join(user_max_product, usingColumn = "user_id")
		}
		// 窗口特征统计
		val window_scope = Array(3, 5, 7, 15, 30)
		
		for (window <- window_scope){
			val time_udf = udf((time:Int) => {math.floor((time-1)/window)})
			var window_agg = full_click_data.withColumn("window_num_"+window.toString, time_udf(col("time")))
			println("window_num=", window)
			window_agg.show(false)
			val feature_names = Array("creative_id", "ad_id", "product_id", "product_category", "advertiser_id", "industry")
			for (feature_name <- feature_names){
			
			}
		}
		
		user_agg
	}
	
	def readAllClickData(sparkSession: SparkSession, dataPath: String, savePath: String, numPartitions: Int): Dataset[Row] = {
		val click_schema = StructType(List(
			StructField("time", IntegerType), StructField("user_id", IntegerType), StructField("creative_id", IntegerType),
			StructField("click_times", IntegerType)
		))
		val user_schema = StructType(List(
			StructField("user_id", IntegerType), StructField("age", IntegerType), StructField("gender", IntegerType)
		))
		
		val train_click_data = sparkSession.read.schema(click_schema).format("csv").option("header", "true")
			.load(dataPath + "/train_preliminary/click_log.csv")
		val user_data = sparkSession.read.schema(user_schema).format("csv").option("header", "true")
			.load(dataPath + "/train_preliminary/user.csv").repartition(numPartitions)
		
		val train_click = train_click_data.join(user_data, usingColumn = "user_id")
		
		val test_click_data = sparkSession.read.schema(click_schema).format("csv").option("header", "true")
			.load(dataPath + "/test/click_log.csv").repartition(numPartitions).withColumn("age", lit(0)).withColumn("gender", lit(0))
		
		val all_click_data = train_click.union(test_click_data).repartition(numPartitions)
		
		val click_user_data = all_click_data
		println("all click data")
		click_user_data.show(50, false)
		println("all click count=", click_user_data.count())
		click_user_data
	}
	
	def readAllAdData(sparkSession: SparkSession, dataPath: String, savePath: String, numPartitions: Int): sql.DataFrame = {
		val schema = StructType(List(
			StructField("creative_id", StringType), StructField("ad_id", StringType), StructField("product_id", StringType),
			StructField("product_category", StringType), StructField("advertiser_id", StringType), StructField("industry", StringType)
		))
//		val train_ad_data = sparkSession.read.schema(schema).format("csv").option("header", "true")
//			.load(dataPath + "/train_preliminary/ad.csv")
//		println("train_ad_data count=", train_ad_data.count())  //2481136L
//		train_ad_data.show(false)
//
//		val test_ad_data = sparkSession.read.schema(schema).format("csv").option("header", "true").load(dataPath + "/test/ad.csv")
//		println("test_ad_data count=", test_ad_data.count())  //2618160L
//		test_ad_data.show(false)
//
//		// creative_id, ad_id, product_id, product_category, advertiser_id, industry
//		val all_ad_data = train_ad_data.union(test_ad_data).repartition(numPartitions)
//			.na.fill(Map("ad_id" -> 4000000, "product_id" -> 60000, "product_category" -> 30,
//			"advertiser_id" -> 63000, "industry" -> 400)).distinct()
		val all_ad_data = sparkSession.read.schema(schema).format("csv").option("header", true)
					.load(dataPath + "/all_ad.csv").repartition(numPartitions)
					.na.fill(Map("ad_id" -> "4000000", "product_id" -> "60000", "product_category" -> "30",
						"advertiser_id" -> "63000", "industry" -> "400"))
		
		println("all ad count=", all_ad_data.count()) // 3412772
		println("all Ad data")
		all_ad_data.show(false)
		all_ad_data
	}
	
	def adTrainFeatureProcess(train_click_data: Dataset[Row], sparkSession: SparkSession, dataPath: String, numPartitions: Int): Unit = {
		
		println("train click data describe")
		train_click_data.describe().show(false)
		//		println("describe")
		// 全部用户的平均点击数：35.679
		
		val K = 2
		val splits = train_click_data.randomSplit(Array(0.5, 0.5), seed = 2020L)
		//		, "product_category", "advertiser_id"
		val feature_names = Array("industry")
		//		, "gender"
		val target_features = Array("age")
		val target_val = Array("click_times")
		
		for (k <- Array.range(0, K)) {
			splits(k) = splits(k).withColumn("fold", lit(k))
			println("split=", k)
			//			splits(k).show(false)
			if (k > 0) {
				splits(0) = splits(0).union(splits(k))
			}
		}
		val fold_data = splits(0).persist(StorageLevel.MEMORY_AND_DISK)
		train_click_data.unpersist()
		var encoded_data: Dataset[Row] = null
		// 五折统计目标编码
		for (k <- Array.range(0, K)) {
			var k_fold = fold_data.filter("fold=" + k)
			println("start process ", k, " fold data")
			k_fold.show(false)
			val other = fold_data.filter("fold!=" + k)
			val encoder = targetEncoder(other, k_fold, feature_names, target_features, target_val, k)
			println("after targetEncoder, res=")
			encoder.show(20, false)
			if (k > 0) {
				encoded_data.union(encoder)
			} else encoded_data = encoder
		}
		println("encoded_data count=", encoded_data.count())
		encoded_data.show(20, false)
		
	}
	
	// 目标编码
	def targetEncoder(other: Dataset[Row], k_fold: Dataset[Row], feature_names: Array[String],
	                  target_features: Array[String], target_vals: Array[String], fold_num: Int): Dataset[Row] = {
		var res = k_fold
		// compute target mean
		// 构造特征：被各age点击的概率，各age的人均点击次数，各age的人均每天点击次数，各age人每天最多点击次数，各age人每天最少点击次数
		for (feature_name <- feature_names) {
			for (target_feature <- target_features) {
				for (target_val <- target_vals) {
					val tmp = other.select(feature_name, target_feature, target_val, "user_id", "time").persist(StorageLevel.MEMORY_AND_DISK)
					println("feature_name=", feature_name, "    target=", target_feature, "    Target_val=", target_val, "   data count=", tmp.count())
					tmp.show(false)
					// feature_name 的点击数总和
					val emperical_df = tmp.groupBy(feature_name).agg(sum(target_val).as(feature_name + "_" + target_val + "_sum"),
						count("user_id").as(feature_name + "_" + "user_id" + "_count"))
						.orderBy(feature_name)
					println("emperical grouped by feature_name=", feature_name)
					emperical_df.show(false)
					
					val grouped_target_df = tmp.groupBy(feature_name, target_feature)
					// 各age的点击次数总和，  求被各age点击的概率使用
					val clicks_sum = grouped_target_df.agg(sum(target_val).as(feature_name + "_" + target_feature + "_" + target_val + "_sum"),
						count("user_id").as(feature_name + "_" + target_feature + "_" + "user_id" + "_count"))
						.orderBy(feature_name, target_feature)
					println("grouped by", feature_name, " and ", target_feature, " clicks sum=")
					clicks_sum.show(false)
					//
					var click_likelihood = clicks_sum.join(emperical_df, usingColumn = feature_name)
					click_likelihood = click_likelihood.withColumn(target_feature + "_click_likelihood",
						click_likelihood(feature_name + "_" + target_feature + "_" + target_val + "_sum") / click_likelihood(feature_name + "_" + target_val + "_sum"))
						.withColumn("user_avg_click",
							click_likelihood(feature_name + "_" + target_feature + "_" + target_val + "_sum") / click_likelihood(feature_name + "_" + target_feature + "_" + "user_id" + "_count"))
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
