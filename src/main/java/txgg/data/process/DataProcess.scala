package txgg.data.process

import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.sql.{DataFrame, Dataset, Row, SQLContext, SparkSession}
import org.apache.spark.sql.types.{ArrayType, FloatType, IntegerType, StringType, StructField, StructType}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.unsafe.types.UTF8String
import org.apache.spark.unsafe.hash.Murmur3_x86_32.hashUnsafeBytes
import org.apache.spark.ml.feature.Word2VecModel
import org.apache.spark.ml.feature.Word2Vec
import java.text.DateFormat
import java.text.SimpleDateFormat
import java.util.Calendar

import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.lit

import scala.util.parsing.json.JSON
import scala.util.Random
import scala.collection.JavaConversions._
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

object DataProcess {
	def main(args: Array[String]): Unit = {
		//        Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
		if (args.length == 0) print("Please set parameter")
		println("args=", args.mkString(", "))
		//        val params = SparkUtil.parse(args)
		//        val dt = params.getOrElse("dt", "20200227")
		val func_name = args(0)
		val dataPath = args(1)
		
		val sparkSession = SparkSession.builder().appName("MaterialTest")
			.config("hive.metastore.warehouse.dir", "")
			.config("spark.sql.warehouse.dir", "hdfs://ns3-backup/user/weibo_bigdata_push/weichao_spark_sql_warehouse")
			.config("hive.exec.scratchdir", "hdfs://ns3-backup/user/weibo_bigdata_push/weichao_hive_scratchdir")
			.enableHiveSupport()
			.getOrCreate()
		val sparkContext = sparkSession.sparkContext
		val sparkConf = sparkContext.getConf
		val sqlContext = new SQLContext(sparkContext)
		val numPartitions = 860
		
		val labelMap = sparkContext.broadcast(collection.Map("00" -> -1, "11" -> 0, "12" -> 1, "21" -> 2, "22" -> 3,
			"31" -> 4, "32" -> 5, "41" -> 6, "42" -> 7,
			"51" -> 8, "52" -> 9, "61" -> 10, "62" -> 11,
			"71" -> 12, "72" -> 13, "81" -> 14, "82" -> 15,
			"91" -> 16, "92" -> 17, "101" -> 18, "102" -> 19))
		println("lableMap=", labelMap.value.mkString(";"))
		val train_data_sql =
			s"""select A.user_id, A.age, A.gender, collect_list(concat_ws("#", time, creative_id, ad_id,
                    product_id,  product_category, advertiser_id, industry, click_times)) as seq
            from (select * from zhongrui3_user_ad_hist_origin where dt='20200507_origin' order by user_id,time) as A
            group by A.user_id,A.age,A.gender limit 1000""".stripMargin
		val predict_data_sql =
			s"""select A.user_id, collect_list(concat_ws("#", time, creative_id, ad_id,
                    product_id,  product_category, advertiser_id, industry, click_times)) as seq
            from (select * from zhongrui3_user_ad_hist_origin where dt='20200507_test' order by user_id,time) as A
            group by A.user_id""".stripMargin
		println("train data_sql=", train_data_sql, "predict_data_sql", predict_data_sql)
		println("funcname=", func_name)
		if (func_name == "newuserlist") {
			newUserList(sparkSession, numPartitions, dataPath + "/new_user_list/")
		} else {
			println("please enter a funcname: uid_ad_info / sequence_uid_ad / makegraph / ad_list")
		}
	}
	def newUserList(sparkSession: SparkSession, numPartitions: Int, dataPath: String): Unit ={
		val data_sql =
			s"""select A.user_id, A.age, A.gender, collect_list(concat_ws("#", cast(time as string), creative_id, ad_id,
                    product_id,  product_category, advertiser_id, industry, cast(click_times as string))) as seq
            from (select * from zhongrui3_user_ad_hist_origin order by user_id,time) as A
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
			.mode("overwrite").save(dataPath + s"/txpredict.tfrecords")
		
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
				.mode("overwrite").save(dataPath + s"/" +  k.toString +"_fold/txtest.tfrecords")
			var train:Dataset[Row] = null
			for (j <- Array.range(0, splits.length)){
				if (j != k){
					train = if (train == null) splits(j)  else train.union(splits(j))
				}
			}
			println("train count=", train.count())
			train.show(false)
			train.repartition(1).write.format("tfrecords").option("recordType", "Example")
				.mode("overwrite").save(dataPath + s"/" +  k.toString +"_fold/txtrain.tfrecords")
		}
		//		//////   train_1  %8
		//		val train_ad_data = data.filter(p => (p._1.toInt % 8 != 0 && p._2.toInt > 0 && p._3.toInt > 0))
		//			.map(p => Row(Array(p._1, p._2, p._3), p._4)) // user_id&label, ad_seq
		//		val train_data_df = sparkSession.createDataFrame(train_ad_data, schema)
		//		println("train result")
		//		train_data_df.show(20, false)
		//		println("train count=", train_ad_data.count())
		//		train_data_df.repartition(2).write.format("tfrecords").option("recordType", "Example")
		//			.mode("overwrite").save(dataPath + s"/txtrain.tfrecords")
		//		//////   test_1  %8
		//		val test_ad_data = data.filter(p => (p._1.toInt % 8 == 0 && p._2.toInt > 0 && p._3.toInt > 0))
		//			.map(p => Row(Array(p._1, p._2, p._3), p._4)) // user_id&label, ad_seq
		//		val test_data_df = sparkSession.createDataFrame(test_ad_data, schema)
		//		println("test result")
		//		test_data_df.show(20, false)
		//		println("test count=", test_ad_data.count())
		//		test_data_df.repartition(2).write.format("tfrecords").option("recordType", "Example")
		//			.mode("overwrite").save(dataPath + s"/txtest.tfrecords")
		
	}
}