import findspark
import pandas as pd
from pyspark.sql import SparkSession
import numpy as np
from pyspark.sql.functions import lit, monotonically_increasing_id, col, array, udf, avg
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, FloatType, IntegerType

# import org.apache.spark.sql._


findspark.init()




def kmeans_fit(data, k, max_iter, q, init):
    centroids = np.array(init)
    for j in range(max_iter):
        prev_cent = centroids
        print(np.linalg.norm(np.array(prev_cent) - np.array(centroids)))
        centroids , tagged_data = runer(data, k, max_iter, q, centroids)
        if np.linalg.norm(np.array(prev_cent) - np.array(centroids)) < 0.0001:
            return centroids
    return centroids


def init_spark(app_name: str):
    spark = SparkSession.builder.appName(app_name).getOrCreate()
    sc = spark.sparkContext
    return spark, sc


def runer(data, k, max_iter, q, init):
    centroids = init
    tagged_data = data.withColumn("id", monotonically_increasing_id()).withColumn("cluster_id", lit(0)).withColumn(
        "dist_from_cent", lit(float('inf')))
    # rdd = data.rdd.map(lambda x: min_argmin(x, centroids)).show()
    maturity_udf = udf(
        lambda a, d=centroids: str(min((np.linalg.norm(a - d[i]), i) for i in range(len(centroids))))[1:-1])

    tagged_data = tagged_data.withColumn("dist_from_cent",
                                         maturity_udf(array(tagged_data._1, tagged_data._2, tagged_data._3)))
    split_col = F.split(tagged_data['dist_from_cent'], ',')
    tagged_data = tagged_data.withColumn('dist', split_col.getItem(0).cast("double")).withColumn('Cluster_num',
                                                                                                 split_col.getItem(
                                                                                                     1).cast("integer"))
    tagged_data = tagged_data.orderBy(col("Cluster_num").asc(), col("dist").asc())
    centroidsTemp = tagged_data.groupBy('Cluster_num').agg(avg("_1").alias("_1"), avg("_2").alias("_2"),
                                                           avg("_3").alias("_3"))
    centroids = np.array(centroidsTemp.select('_1', '_2', '_3').collect())
    centroids[[0, 1]] = centroids[[1, 0]]
    print(centroids)
    return centroids, tagged_data


def main():
    spark, sc = init_spark('RoiNYarin')
    data_file = spark.read.parquet("random_data.parquet")
    # data_file.toPandas()
    k = 2
    init = data_file.rdd.take(2)
    # df = pd.read_parquet("random_data.parquet")
    kmeans_fit(data_file, 2, k, 2, init)


if __name__ == "__main__":
    main()
