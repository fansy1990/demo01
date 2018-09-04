package kmeans_test
//import breeze.linalg.*
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._

class MapKMeansID2NameTransformer( override val uid:String) extends Transformer {
  final val inputCol= new Param[String](this, "inputCol", "The input column")
  final val outputCol = new Param[String](this, "outputCol", "The output column")
  final val indexLabelId = new Param[String](this,"indexLabelid","the index with label string")

  def setInputCol(value: String): this.type = set(inputCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  def setIndexLabelId(value:String): this.type =set(indexLabelId,value)



  def this() = this(Identifiable.randomUID("MapKMeansID2NameTransformer"))


  override def transformSchema(schema: StructType): StructType = {
    // Check that the input type is a string
    val idx = schema.fieldIndex($(inputCol))
    val field = schema.fields(idx)
    if (field.dataType != IntegerType) {
      throw new Exception(s"Input type ${field.dataType} did not match input type IntegerType")
    }
    // Add the return field
    schema.add(StructField($(outputCol), IntegerType, false))
  }

  def transform(df: DataFrame): DataFrame = {
    val index_labelId = ($(indexLabelId)).split(":").map{x => val t= x.split(","); (t(0).toInt,t(1).toInt)}.toMap
    val mapId2LabelId = udf { in: Int => index_labelId.get(in).getOrElse(-1) }
    //    df.select(col("*"),
    //      mapId2LabelId(df($(inputCol))).as($(outputCol))
    //    )
    df.withColumn($(outputCol), mapId2LabelId(col($(inputCol))))
  }

  override def copy(extra: ParamMap): Transformer =defaultCopy(extra)

}