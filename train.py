import tensorflow.keras.datasets.mnist as input_data

(x_train, y_train), (x_test, y_test) = input_data.load_data()

x_train = x_train.reshape((-1 , 28*28 ))
x_test = x_test.reshape((-1 , 28*28 ))

from kingdra_cluster.kingdra_cluster import KingdraCluster
model = KingdraCluster( n_iter=5 )
model.fit( x_train )
model.save_weights('./model.trained')