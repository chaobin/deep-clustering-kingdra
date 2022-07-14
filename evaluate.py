import tensorflow.keras.datasets.mnist as input_data
from kingdra_cluster.kingdra_cluster import KingdraCluster
from kingdra_cluster.unsup_metrics import ACC

(x_train, y_train), (x_test, y_test) = input_data.load_data()

x_train = x_train.reshape((-1 , 28*28 ))
x_test = x_test.reshape((-1 , 28*28 ))

model = KingdraCluster( n_iter=5 )
model.load_weights('./model.trained')
preds = model.predict( x_test )

print("Accuracy: " , ACC( y_test  ,  preds ) )