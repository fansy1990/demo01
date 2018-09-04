package kmeans_test

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}

/**
  * //@Author: fansy 
  * //@Time: 2018/9/3 10:41
  * //@Email: fansy1990@foxmail.com
  */
object CommonUtils {
  val featureCols="l,r,f,m,c"
  val phone_no="phone_no"
//  val tv_path = "C:\\Users\\fansy\\tmp\\demo01\\src\\data\\tv_kmeans.csv"
  //val tv_path = "C:\\Users\\fansy\\tmp\\demo01\\src\\data\\bd_kmeans.csv"
  val tv_path = "C:\\Users\\fansy\\tmp\\demo01\\src\\data\\bd_9_kmeans.csv"
  def getDF(sc:SparkContext,sqlContext:SQLContext ): DataFrame ={
    val schema =
      StructType(
        featureCols.split(",").map(fieldName => StructField(fieldName.trim, DoubleType, false)) :+
          StructField(phone_no,StringType,false))
    val rdd :RDD[Row]= sc.textFile(tv_path).map{x => val t = x.split(","); Row.merge( Row.fromSeq(t.tail.map(_.toDouble)),Row.fromSeq(Seq(t(0))))}
    sqlContext.createDataFrame(rdd,schema)
  }

}

case class lrfmc(phone_no:String , l:Double,r:Double,f:Double,m:Double,c:Double)