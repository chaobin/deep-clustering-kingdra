import numpy as np
import ipdb
import os
import json
import sys
from kingdra_cluster.unsup_metrics import ACC
from kingdra_cluster.kingdra_cluster import KingdraCluster
import warnings
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

warnings.filterwarnings("ignore", category=DeprecationWarning)


config = json.loads(open(sys.argv[1]).read())

NAME_TYPE = os.path.basename(os.path.dirname(config.get('training_path', '')))

gt = np.load(config['ground_truth_path'])
X = np.load(config['training_path'])

del config['ground_truth_path']
del config['training_path']


m = KingdraCluster(**config)


def callback(it, y_pred_e, models):
    print("Clustering accuracy at  iteration ", it, ACC(gt,  y_pred_e))


m.load_weights(f"./trained/{NAME_TYPE}.trained")
preds_2 = m.predict(X)

ipdb.set_trace()

print("Clustering Accuracy: ", ACC(gt,  preds_2))
