package kmeans_test

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}

/**
  * //@Author: fansy 
  * //@Time: 2018/9/3 10:23
  * //@Email: fansy1990@foxmail.com
  */
object KMeansTest {

  val features ="features"
  val scaledFeatures="scaledFeatures"
  val phone_no ="phone_no"
  val predictCol = "predictCol"

  def main(args: Array[String]): Unit = {
    println("====")
    //[{"rules":[{"id":74040,"createTime":1525338755419,"ruleName":"c","ruleValue":"","compare":"min","andOr":"AND"}],"label_name":"低价值用户","label_id":26963},{"rules":[{"id":74039,"createTime":1525337056886,"ruleName":"c","ruleValue":"","compare":"max","andOr":"AND"}],"label_name":"重要保持用户","label_id":26959},{"rules":[{"id":41022,"createTime":1524125823458,"ruleName":"m","ruleValue":"","compare":"max","andOr":"AND"}],"label_name":"重要发展用户","label_id":26960},{"rules":[],"label_name":"挽留用户","label_id":194057},{"rules":[{"id":41023,"createTime":1524125856736,"ruleName":"l","ruleValue":"","compare":"max","andOr":"AND"}],"label_name":"一般用户","label_id":26961}]
    val rule_bd = "26963,低价值,c:min;26959,保持,c:max;26960,发展,m:max;194057,挽留,;26961,一般,l:max"
    val rule_tv = "26946,重要保持用户,f:max|c:max;26947,重要发展用户,m:max;26949,一般用户,l:max;26948,重要挽留用户,;26950,低价值用户,f:min|c:min"
    val (input, rules,k,maxIterations,runs,initMode,featureCols,
    output_center, output_every_point) = ("a",rule_bd,5,200,2,"","l,r,f,m,c","c","d")
    val seed = 10L
    // 1. 读取数据，直接构造表，从Hive读取
    val sc: SparkContext = new SparkContext(new SparkConf().setMaster("local[2]").setAppName("kmeanns"))

    val sqlContext :SQLContext = new SQLContext(sc)

    // 1. 读取数据
    val df = CommonUtils.getDF(sc,sqlContext)
    df.show(3)
    // 2. 对数据进行归一化
    val assembler = new VectorAssembler()
      .setInputCols(featureCols.split(",").map(_.trim))
      .setOutputCol(features)

    val assembled_data = assembler.transform(df)
    assembled_data.show(3)
    print("====")
    // 2.1 归一化
    val scaler = new StandardScaler()
      .setInputCol(features)
      .setOutputCol(scaledFeatures)
      .setWithStd(true)
      .setWithMean(true)
    // 归一化后的数据
    val scaled_data = scaler.fit(assembled_data).transform(assembled_data)
    println("===")
    scaled_data.select("features","scaledFeatures").show(10,false)
    scaled_data.cache()

    // 3. 聚类
//    val vec_data = id_vec.map(_._1)
//    vec_data.cache()
      val kmeans_model = new KMeans().setFeaturesCol(scaledFeatures).setK(5).setMaxIter(10).setInitMode("k-means||").setInitSteps(10).setTol(0.5).setPredictionCol(predictCol).fit(scaled_data)
    kmeans_model.clusterCenters.foreach(f => println(f.toDense.values.mkString(",")))
//    val clusters = KMeans.train(vec_data, k, maxIterations,runs,initMode,seed)
    val result = kmeans_model.transform(scaled_data)

    result.select(predictCol).groupBy(predictCol).agg(count(predictCol)).show(10,false)

    result.show(3,false)

    val center_label_name_id_index =
      getClusterLabelName(kmeans_model.clusterCenters,rules,featureCols)

    // 4. 聚类结果构造DataFrame
    val center_label_name_df = constructDataFrame(sc,sqlContext,center_label_name_id_index,featureCols)
    center_label_name_df.show(5,false)
    // 5. 对原始数据进行分类
    val index_labelId = center_label_name_id_index.map(x => (x._4+","+x._3)).mkString(":")
    val mapKMeansID2NameTransformer = new MapKMeansID2NameTransformer( )
      .setInputCol("predictCol").setOutputCol("label_id")
      .setIndexLabelId(index_labelId)
//    val predict_data = kmeans_model.transform(scaled_data)
    val transformed_data = mapKMeansID2NameTransformer.transform(result).select(phone_no,"label_id")

    transformed_data.show(10,false)


    sc.stop()
  }

  def getClusterLabelName(clusterCenters: Array[Vector], rulesStr: String, featureCols: String) = {
    val rules = rulesStr.split(";").map(_.trim) //
    val center_label_name_id =
      for(rule <- rules ) yield getLabelNameId(rule,featureCols,clusterCenters)
    // 1. 找到输出下标为-1的记录
    val nullIndex = center_label_name_id.zipWithIndex.filter(x => x._1._4 == -1).apply(0)
    // 2. 找到 clusterCenters中为空的记录
    val realIndex = Range(0,center_label_name_id.length).toSet.diff(center_label_name_id.map(x => x._4).toSet).toList.apply(0)
    center_label_name_id.update(nullIndex._2,
      (clusterCenters.apply(realIndex),nullIndex._1._2,nullIndex._1._3,realIndex))
    center_label_name_id
  }

  def getLabelNameId(rule: String, featureCols: String, clusterCenters: Array[Vector]):
  (Vector,String,Int,Int) = {
    val ruleArr = rule.split(",",-1).map(_.trim)
    val(label_id, label_name, label_rules) = (ruleArr(0).toInt,ruleArr(1),ruleArr(2))
    if(label_rules.size < 1){
      return (null,label_name,label_id,-1)
    }
    val label_rulesArr = label_rules.split("\\|").map(_.trim)
    val all_rule_indexes =for(label_rule <- label_rulesArr) yield getRuleIndex(label_rule,featureCols,clusterCenters)
    if(all_rule_indexes.forall(x => x == all_rule_indexes.apply(0))){
      //      println((clusterCenters.apply(all_rule_indexes.apply(0)),label_name,label_id))
      (clusterCenters.apply(all_rule_indexes.apply(0)),label_name,label_id, all_rule_indexes.apply(0))
    }else{
      throw new RuntimeException("rules give not only labelName and labelId \n"+
        rule+"\n"+all_rule_indexes.mkString(","))
    }
  }

  def getRuleIndex(label_rule: String, featureCols: String, clusterCenters: Array[Vector]) :Int= {
    val arr  = label_rule.split(":").map(_.trim)
    val (col,comparator) = (arr(0),arr(1))
    val index = featureCols.split(",").map(_.trim).indexOf(col)
    val columnData =clusterCenters.map(_.apply(index))
    comparator match {
      case "max" =>
        columnData.indexOf(columnData.max)
      case "min" => columnData.indexOf(columnData.min)
      case _ => throw new RuntimeException(" not support comparator!")
    }
  }
  def constructDataFrame(sc:SparkContext,sqlContext: SQLContext, center_label_name_id: Array[(Vector, String, Int,Int)], featureCols: String) = {
    val schema =
      StructType(
        featureCols.split(",").map(fieldName => StructField(fieldName.trim, DoubleType, false)) :+
          StructField("label_name",StringType,false))
    val rdd = sc.parallelize(center_label_name_id.map(x =>
      Row.merge(Row.fromSeq(x._1.toArray ), Row.fromSeq(Seq(x._2))) ))
    sqlContext.createDataFrame(rdd,schema)
  }

}
