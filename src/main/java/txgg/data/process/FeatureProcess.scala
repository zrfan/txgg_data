package txgg.data.process

import com.microsoft.ml.spark.lightgbm.LightGBMClassifier
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql
import org.apache.spark.sql.functions.{approx_count_distinct, col, count, lit, sum, udf, mean, max, min}
import org.apache.spark.sql.types.{ArrayType, IntegerType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SQLContext, SparkSession}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.StatCounter

import scala.collection.mutable
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
		val numPartitions = 460
		val dataPath = "/home/fzr/txgg/data/origin/"
		val savePath = "/home/fzr/txgg/data/processed/"
		println("dataPath=", dataPath)
		println("funcname=", func_name)
		val all_ad_data = readAllAdData(sparkSession, dataPath, savePath, numPartitions).persist(StorageLevel.MEMORY_AND_DISK)
		println("all_ad data=")
		all_ad_data.show(50, false) // 去重后广告数3412773
		println("all_ad_data count=", all_ad_data.count())
		
		val all_click_data = readAllClickData(sparkSession, dataPath, savePath, numPartitions).persist(StorageLevel.MEMORY_AND_DISK) // user click age&gender
		println("all_click_data count=", all_click_data.count())
		val tmp = all_click_data.filter("time>91")
		println("click time>91")
		tmp.show(false)
		
		println("all_ad_click_data_before_join")
		all_ad_data.show(50, false)
		all_click_data.show(50, false)
		
		val full_click_data = all_click_data.join(all_ad_data, usingColumns = Seq("creative_id"), joinType = "left_outer")
			.repartition(numPartitions)
			.persist(StorageLevel.MEMORY_AND_DISK)
		all_click_data.unpersist()
		all_ad_data.unpersist()
		println("full click data after join")
		full_click_data.show(50, false)
		if (func_name == "newuserlist"){
			newUserList(full_click_data, sparkSession, numPartitions, savePath)
		}else if (func_name == "featuretest"){
			featureTest(full_click_data, sparkSession, dataPath, savePath, numPartitions)
		}
	}
	
	def featureTest(full_click_data: Dataset[Row], sparkSession: SparkSession, dataPath: String, savePath: String, numPartitions: Int): Unit ={
		// 用户特征提取
		val user_feature = userFeatureProcess(full_click_data, sparkSession, savePath, numPartitions).persist(StorageLevel.MEMORY_AND_DISK)
		println("user_feature")
		user_feature.show(false)
		
		val all_feature_cols = Array("all_click_cnt", "active_days", "active_avg_clicks",
			"creative_cnt", "ad_cnt", "product_cnt", "category_cnt", "advertiser_cnt", "industry_cnt",
			"mean_dur", "max_dur", "min_dur", "mean_dur2", "max_dur2", "min_dur2", "stdev", "variance", "popStdev", "sampleStdev", "popVariance", "sampleVariance",
			"max_click_product_id", "max_click_product_category", "max_click_advertiser_id", "max_click_industry",
			"window3_click_times_avg")
		
		val all_data = user_feature.select((all_feature_cols ++ Array("user_id", "age", "gender")).map(x => col(x)): _*).persist(StorageLevel.MEMORY_AND_DISK)
		println("all_data")
		all_data.show(200, false)
		user_feature.unpersist()
		
		val assembler = new VectorAssembler().setInputCols(all_feature_cols).setOutputCol("features")
		val all_assembled_data = assembler.transform(all_data)
		println("assembled")
		all_assembled_data.show(false)
		val all_train = all_assembled_data.filter("age!=0 and gender!=0")
			.withColumn("label_gender", user_feature("gender") * 1.0 - 1.0)
			.withColumn("label_age", user_feature("age") * 1.0 - 1.0)
		val Array(train, test) = all_train.randomSplit(Array(0.7, 0.3), seed = 2020L)
		// train gender
		val lightgbm_gender = new LightGBMClassifier().setLabelCol("label_gender").setFeaturesCol("features")
			.setPredictionCol("predict_gender").setProbabilityCol("gender_probability")
		
		val model_gender = lightgbm_gender.fit(train)
		val val_gender = model_gender.transform(test)
		println("val_gender=", val_gender)
		val_gender.show(false)
		val evaluator_gender = new BinaryClassificationEvaluator().setLabelCol("label_gender").setRawPredictionCol("predict_gender")
		println("gender evalutor=", evaluator_gender.evaluate(val_gender))
		
		// train age
		val lightgbm_age = new LightGBMClassifier().setLabelCol("label_age").setFeaturesCol("features")
			.setPredictionCol("predict_age").setProbabilityCol("age_probability")
		val model_age = lightgbm_age.fit(train)
		val val_age = model_age.transform(test)
		println("val_age=", val_age.show(false))
		val evaluator_age = new MulticlassClassificationEvaluator().setLabelCol("label_age").setPredictionCol("predict_age")
		println("gender evalutor=", evaluator_age.evaluate(val_age))
		
		
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
		// 全部用户的平均点击数：35.679
		full_click_data.createTempView("txgg_temp")
//		 collect_list(creative_id) as creative_list, collect_list(ad_id) as ad_list,
// collect_set(product_id) as product_set, collect_set(product_category) as category_set,
// collect_set(advertiser_id) as advertiser_set, collect_set(industry) as industry_set
		val user_agg_sql =
			s"""select user_id, age, gender, sum(click_times) as all_click_cnt, count(distinct time) as active_days,
			   | count(distinct creative_id) as creative_cnt,
			   | count(distinct ad_id) as ad_cnt, count(distinct product_id) as product_cnt,
			   | count(distinct product_category) as category_cnt, count(distinct advertiser_id) as advertiser_cnt,
			   | count(distinct industry) as industry_cnt,
			   | collect_list(time) as time_list
			   | from (select * from txgg_temp order by time) as A group by user_id, age, gender """.stripMargin
		var user_agg = sparkSession.sql(user_agg_sql)
		println("user_agg info")
		user_agg = user_agg.withColumn("active_avg_clicks", user_agg("all_click_cnt") * 1.0 / user_agg("active_days"))
		
		// 点击间隔统计
		def getDuring(time_list: scala.collection.mutable.WrappedArray[Int]): Array[Double] = {
			val dur_list = time_list.map(x => x.toFloat).sorted.sliding(2).map(x => x.last - x.head).toList
			val mean_dur = dur_list.sum / dur_list.length
			val max_dur = dur_list.max
			val min_dur = dur_list.min
			val stats = StatCounter()
			for(x <- dur_list) stats.merge(x)
			Array(mean_dur, max_dur, min_dur, stats.mean, stats.max, stats.min, stats.stdev, stats.variance,
				stats.popStdev, stats.sampleStdev, stats.popVariance, stats.sampleVariance)
		}
		
		val durUDF = udf((time_list: scala.collection.mutable.WrappedArray[Int]) => {
			getDuring(time_list)
		})
		val user_dur = user_agg
			.withColumn("dur", durUDF(col("time_list")))
			.select(col("user_id"), col("dur").getItem(0).as("mean_dur"),
				col("dur").getItem(1).as("max_dur"),
				col("dur").getItem(2).as("min_dur"),
				col("dur").getItem(3).as("mean_dur2"),
				col("dur").getItem(4).as("max_dur2"),
				col("dur").getItem(5).as("min_dur2"),
				col("dur").getItem(6).as("stdev"),
				col("dur").getItem(7).as("variance"),
				col("dur").getItem(8).as("popStdev"),
				col("dur").getItem(9).as("sampleStdev"),
				col("dur").getItem(10).as("popVariance"),
				col("dur").getItem(11).as("sampleVariance"))
		println("test mean dur")
		user_dur.show(false)
		user_agg = user_agg.join(user_dur, usingColumn = "user_id")
		// 最大点击特征统计
		val max_feature_names = Array("product_id", "product_category", "advertiser_id", "industry")
		for (name <- max_feature_names) {
			val user_max_click_sql =
				s"""select b.user_id, b.$name as max_click_$name, b.cnt from (
				   |    select user_id, $name, cnt, row_number() over (partition by user_id order by cnt desc) rank
				   |    from ( select user_id, $name, sum(click_times) as cnt from txgg_temp group by user_id, $name) a
				   |  ) b where b.rank=1""".stripMargin
			val user_max_product = sparkSession.sql(user_max_click_sql)
//			println("user_max_click data")
//			user_max_product.show(false)
			user_agg = user_agg.join(user_max_product, usingColumn = "user_id")
		}
		// 窗口特征统计 , 3, 5, 7, 15, 30
		val window_scope = Array(7, 30)
		
		for (window <- window_scope) {
			val time_udf = udf((time: Int) => {
				math.floor((time - 1) / window)
			})
			var window_df = full_click_data.withColumn("window_num_" + window.toString, time_udf(col("time")))
			println("window_num=", window)
			window_df.show(false)
			
			val window_agg = window_df.groupBy("user_id", "window_num_" + window.toString)
			val click_times = window_agg.agg(sum("click_times").as("window"+window+"_click_times"))
			val window_mean_click = click_times.groupBy("user_id")
				.agg(mean("window"+window+"_click_times").as("window"+window+"_click_times_avg"))
			window_mean_click.show(20, false)
			user_agg = user_agg.join(window_mean_click, usingColumn = "user_id")
			
			val feature_names = Array("creative_id", "ad_id", "product_id", "product_category", "advertiser_id", "industry")
			for (feature_name <- feature_names) {
				val window_res = window_agg.agg(approx_count_distinct(feature_name).as(feature_name+"_window"+window+"_nunique"))
				println("window_agg=", feature_name, " window_num="+window.toString)
				window_res.show(20, false)
				val user_window_agg = window_res.groupBy("user_id")
				val user_res = user_window_agg.agg(mean(feature_name+"_window"+window+"_nunique").as(feature_name+"_window"+window+"_nunique_avg"))
				user_res.show(20, false)
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
		val schema = StructType(List(
			StructField("time", IntegerType), StructField("user_id", IntegerType), StructField("creative_id", IntegerType),
			StructField("click_times", IntegerType), StructField("age", IntegerType), StructField("gender", IntegerType)
		))
		
		val train_click_data = sparkSession.read.schema(click_schema).format("csv").option("header", true)
			.load(dataPath + "/train_preliminary/click_log.csv")
		val user_data = sparkSession.read.schema(user_schema).format("csv").option("header", true)
			.load(dataPath + "/train_preliminary/user.csv").repartition(numPartitions)
		
		val train_click = train_click_data.join(user_data, usingColumn = "user_id").rdd
			.map(p => (p.getAs[String]("time"), p.getAs[String]("user_id"),
				p.getAs[String]("creative_id"), p.getAs[String]("click_times"),
				p.getAs[String]("age"), p.getAs[String]("gender")))
		train_click.take(50).foreach(p => println("train click data=", "time=",p._1, "user_id=", p._2,
			"creative_id=", p._3, "click_times=", p._4, "age=", p._5, "gender=", p._6))
		
		val test_click_data = sparkSession.read.schema(click_schema).format("csv").option("header", true)
			.load(dataPath + "/test/click_log.csv").repartition(numPartitions)
			.withColumn("age", lit(0)).withColumn("gender", lit(0)).rdd
			.map(p => (p.getAs[String]("time"), p.getAs[String]("user_id"),
				p.getAs[String]("creative_id"), p.getAs[String]("click_times"),
				p.getAs[String]("age"), p.getAs[String]("gender")))
		test_click_data.take(50).foreach(p => println("test click data=", "time=",p._1, "user_id=", p._2,
			"creative_id=", p._3, "click_times=", p._4, "age=", p._5, "gender=", p._6))
		
		val all_click_data = train_click.union(test_click_data).repartition(numPartitions)
		
		all_click_data.take(50).foreach(p => println("all click data=", "time=",p._1, "user_id=", p._2,
			"creative_id=", p._3, "click_times=", p._4, "age=", p._5, "gender=", p._6))
		
		println("all click count=", all_click_data.count()) // 63668283
		val res = all_click_data.map(p => Row(p._1, p._2, p._3, p._4, p._5, p._6))
		sparkSession.createDataFrame(res, schema)
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
		val value_map = Map("ad_id" -> "4000000", "product_id" -> "60000", "product_category" -> "30",
			"advertiser_id" -> "63000", "industry" -> "400")
		val all_ad_data = sparkSession.read.schema(schema).format("csv").option("header", true)
			.load(dataPath + "/all_ad.csv").repartition(numPartitions).rdd
			.map(p => (p.getAs[String]("creative_id"), p.getAs[String]("ad_id"), p.getAs[String]("product_id"),
				p.getAs[String]("product_category"), p.getAs[String]("advertiser_id"), p.getAs[String]("industry")))
			.map(p => (p._1, if (p._2 == "\\N") "4000000" else p._2,
				if (p._3 == "\\N") "60000" else p._3,
				if (p._4 == "\\N") "30" else p._4,
				if (p._5 == "\\N") "63000" else p._5,
				if (p._6 == "\\N") "400" else p._6)).map(p => Row(p._1.toInt, p._2.toInt, p._3.toInt, p._4.toInt, p._5.toInt, p._6.toInt))
		val new_schema = StructType(List(
			StructField("creative_id", IntegerType), StructField("ad_id", IntegerType), StructField("product_id", IntegerType),
			StructField("product_category", IntegerType), StructField("advertiser_id", IntegerType), StructField("industry", IntegerType)
		))
		val res = sparkSession.createDataFrame(all_ad_data, new_schema)
		
		println("all ad count=", all_ad_data.count()) // 3412772
		println("all Ad data")
		res.show(false)
		res
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
	
	def newUserList(full_click_data: Dataset[Row], sparkSession: SparkSession, numPartitions: Int, savePath: String): Unit ={
		full_click_data.createTempView("txgg_temp")
		val data_sql =
			s"""select cast(A.user_id as string), cast(A.age as string), cast(A.gender as string),
	            collect_list(concat_ws("#", cast(time as string), cast(creative_id as string), cast(ad_id as string),
                    cast(product_id as string),  cast(product_category as string),
                    cast(advertiser_id as string), cast(industry as string), cast(click_times as string))) as seq
            from (select * from txgg_temp order by user_id,time) as A
            group by A.user_id,A.age,A.gender""".stripMargin
		def getUserSeq(list: Array[String]): Array[String] ={
			// time, creative_id, ad_id, product_id, product_category, advertiser_id, industry, click_times
			val time_ad_list = list.map(x => x.split("#"))
			var res:Array[String] = Array[String]()
			for (i <- Array.range(1, 7)){
				var interest_list: Array[String] = null
				if (i<3){  // creative_id, ad_id
					interest_list = time_ad_list.map(x => (x(i) + "#" + x(7)))
				}else{
					val interest = time_ad_list.map(x => (x(i), x(7).toInt)).groupBy(_._1).mapValues(seq => seq.reduce { (x, y) => (x._1, x._2 + y._2) })
					interest_list = interest.mapValues(seq => seq._1+"#"+seq._2.toString).values.toArray
				}
				if (interest_list.length > 64){  // 超过64长度的截断，保留最后 的64次点击
					interest_list = interest_list.slice(interest_list.length-64, interest_list.length)
				}
				val cnt = interest_list.length
				// 第一位是seq长度，之后是id+'#'+点击次数
				val str = cnt.toString + ";" + interest_list.mkString(";")
				res = res :+ str
			}
			res
		}
		val data = sparkSession.sql(data_sql).rdd.repartition(numPartitions)
			.map(p => (p(0).asInstanceOf[String], p(1).asInstanceOf[String], p(2).asInstanceOf[String],
				p(3).asInstanceOf[mutable.WrappedArray[String]].toArray))
			.map(p => (p._1, p._2, p._3, getUserSeq(p._4)))
			.map(p => (p._1, p._2, p._3, p._4))
			.persist(StorageLevel.MEMORY_AND_DISK)
		val schema = StructType(List(
			StructField("user_id_label", ArrayType(StringType)), StructField("ad_seq", ArrayType(StringType))
		))
		/////	  predict 先写出内存
		val predict_ad_data = data.filter(p => (p._2.toInt == 0 && p._3.toInt == 0))
			.map(p => Row(Array(p._1, p._2, p._3), p._4)) // user_id&label, ad_seq
		val predict_ad_df = sparkSession.createDataFrame(predict_ad_data, schema)
		println("predict result")
		predict_ad_df.show(20, false)
		println("predict count=", predict_ad_df.count())
		predict_ad_df.repartition(2).write.format("tfrecords").option("recordType", "Example")
			.mode("overwrite").save(savePath + s"/txpredict.tfrecords")
		
		val all_train_data = data.filter(p => (p._2.toInt > 0 && p._3.toInt > 0))
			.map(p => Row(Array(p._1, p._2, p._3), p._4)).persist(StorageLevel.MEMORY_AND_DISK) // user_id&label, ad_seq
		val all_train_df = sparkSession.createDataFrame(all_train_data, schema)
		val splits = all_train_df.randomSplit(Array(0.2, 0.2, 0.2, 0.2, 0.2), seed = 2020L)
		data.unpersist()
		for (k <- Array.range(0, splits.length)){
			//			splits(k) = splits(k).withColumn("fold", lit(k))
			println("process_split=", k, "test count=", splits(k).count())
			splits(k).show(false)
			splits(k).repartition(1).write.format("tfrecords").option("recordType", "Example")
				.mode("overwrite").save(savePath + s"/" +  k.toString +"_fold/txtest.tfrecords")
			var train:Dataset[Row] = null
			for (j <- Array.range(0, splits.length)){
				if (j != k){
					train = if (train == null) splits(j)  else train.union(splits(j))
				}
			}
			println("train count=", train.count())
			train.show(false)
			train.repartition(1).write.format("tfrecords").option("recordType", "Example")
				.mode("overwrite").save(savePath + s"/" +  k.toString +"_fold/txtrain.tfrecords")
		}
	}
}
