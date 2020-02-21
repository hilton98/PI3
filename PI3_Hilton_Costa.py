import findspark
findspark.init()
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import Tokenizer,StopWordsRemover,CountVectorizer,MinHashLSH,IDF
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
from pyspark.ml.linalg import *

spark = SparkSession.builder.appName("wc").config("spark.some.config.option", "some-value").getOrCreate()
sc = ("spark.SparkContext")

#path para os dados de teste e treino (no mesmo diretorio do codigo)
pathTrain = ["dataset_train.csv"]
pathTest  = ["dataset_test.csv"]

#preparacao dos dataframes de treino e teste
data_treino = spark.read.load(pathTrain, format="csv",header=True)		#dataset treino
data_test = spark.read.load(pathTest, format="csv",header=True)			#dataset test
print("	Dados de treino")
data_treino.select("*").show()

#declacao de stopwords, tokenizacao, idf, formacao do vocabulario  
tk = Tokenizer(inputCol="Conteudo", outputCol="tokens")
swr = StopWordsRemover(inputCol="tokens", outputCol="words")
cv = CountVectorizer(inputCol="words",outputCol="rawFeatures",vocabSize=100000)
idf = IDF(inputCol="rawFeatures", outputCol="features")

#pipeline dos processos declarados para os dados de teste e treino
pipeline = Pipeline(stages=[tk,swr,cv,idf])
model_pipe = pipeline.fit(data_treino)
data_treino = model_pipe.transform(data_treino)

model_pipe = pipeline.fit(data_test)
data_test = model_pipe.transform(data_test)

#Geracao do modelo e teste
mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=5)
model = mh.fit(data_treino)
data_treino = model.transform(data_treino)
data_treino.show()
#Modelo de dados treinado
'''
te = data_test.select("features").collect()
tr = data_treino.select("features").collect()
'''
data_test.select("features").show()
dadosTef = data_test.select("features").rdd.flatMap(lambda x:x).collect()
print("	Features dos dados de teste")
dadosTr = data_treino.select("NewsGroup","features").rdd.flatMap(lambda x:x).collect()

#model.approxNearestNeighbors(SparseVector(str(tr)),SparseVector(str(te[4])),2),show()

data_test = model.transform(data_test)

# METRICAS PARA AVALICACAO
prediction = model.transform(data_test)

#precision
aval_p = MulticlassClassificationEvaluator(predictionCol='prediction', metricName="weightedPrecision")
precision = aval_p.evaluate(prediction).show()

#Recall
aval_r = MulticlassClassificationEvaluator(predictionCol='prediction', metricName="weightedRecall")
Recall = aval_r.evaluate(prediction).show()

#F1
aval_f1 = MulticlassClassificationEvaluator(predictionCol='prediction', metricName="f1")
f1 = aval_f1.evaluate(prediction).show()
