import findspark
import pandas as pd
from pyspark.sql import SparkSession
import numpy as np
from pyspark.sql.functions import lit, monotonically_increasing_id, col, array, udf, avg
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, FloatType, IntegerType

# import org.apache.spark.sql._


findspark.init()


def sumExmaple(a, b, c, d):
    # print("hi")
    point = np.array([a, b, c])

    return 10


def kmeans_fit(data, k, max_iter, q, init):
    find_distance_udf = udf(sumExmaple)
    centroids = init
    tagged_data = data.withColumn("id", monotonically_increasing_id()).withColumn("cluster_id", lit(0)).withColumn(
        "dist_from_cent", lit(float('inf')))
    # rdd = data.rdd.map(lambda x: min_argmin(x, centroids)).show()
    tagged_data.printSchema()
    tagged_data.show()
    # tagged_data.withColumn("cluster_id",find_distance_udf(array(col("_1"),col("_2"),col("_3"), col("id"), col("dist")))).show()
    maturity_udf = udf(
        lambda a, d=centroids: str(min((np.linalg.norm(a - d[i]), i) for i in range(len(centroids))))[1:-1])
    order = udf(lambda a: a)

    tagged_data = tagged_data.withColumn("dist_from_cent",
                                         lit(maturity_udf(array(tagged_data._1, tagged_data._2, tagged_data._3))))
    split_col = F.split(tagged_data['dist_from_cent'], ',')
    tagged_data = tagged_data.withColumn('dist', split_col.getItem(0).cast("double")).withColumn('Cluster_num',
                                                                                                 split_col.getItem(
                                                                                                     1).cast("integer"))
    tagged_data = tagged_data.orderBy(col("Cluster_num").asc(), col("dist").asc())
    centroidsTemp= tagged_data.groupBy('Cluster_num').agg(avg("_1").alias("_1"), avg("_2").alias("_2"), avg("_3").alias("_3"))
    centroids=np.array(centroidsTemp.select('_1', '_2', '_3').collect())
    # df = tagged_data.withColumn('NAME2', split_col.getItem(1))
    # df = tagged_data.select('id', F.split('dist_from_cent', ',  ').alias('dist_from_cent')).show()
    # df = tagged_data.select('id', 'dist_from_cent').show()
    # tagged_data.withColumn("cluster_id", order(tagged_data.dist_from_cent)).show()
    # tagged_data.filter()


def init_spark(app_name: str):
    spark = SparkSession.builder.appName(app_name).getOrCreate()
    sc = spark.sparkContext
    return spark, sc


def main():
    spark, sc = init_spark('RoiNYarin')
    data_file = spark.read.parquet("random_data.parquet")
    # data_file.toPandas()
    centroids = [np.array([3.14, 3.11, 3.9]), np.array([2.01, 4.55, 1.224])]
    k = 2
    df = pd.read_parquet("random_data.parquet")
    print("hi")
    kmeans_fit(data_file, 2, 13, 2, centroids)


if __name__ == "__main__":
    main()
