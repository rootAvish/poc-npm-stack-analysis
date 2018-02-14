import sys
sys.path.append("..")
from lib.cvae import *
import numpy as np
import tensorflow as tf
import scipy.io
from lib.utils import *

np.random.seed(0)
tf.set_random_seed(0)
init_logging("cvae-packagedata-5.log")

def load_cvae_data():
  data = {}
  data_dir = "../"
  # variables = scipy.io.loadmat(data_dir + "mult_nor.mat")
  # data["content"] = variables['X']

  content_matrix_file = open('../content_matrix.npy', 'rb')
  d = np.load(content_matrix_file)
  content_matrix_file.close()
  # d = d.T
  data["content"] = d

  data["train_users"] = load_rating(data_dir + "packagedata-train-5-users.dat")
  data["train_items"] = load_rating(data_dir + "packagedata-train-5-items.dat")
  data["test_users"] = load_rating(data_dir + "packagedata-test-5-users.dat")
  data["test_items"] = load_rating(data_dir + "packagedata-test-5-items.dat")

  return data

def load_rating(path):
  arr = []
  for line in open(path):
    a = line.strip().split()
    if a[0]==0:
      l = []
    else:
      l = [int(x) for x in a[1:]]
    arr.append(l)
  return arr

params = Params()
params.lambda_u = 0.1
params.lambda_v = 10
params.lambda_r = 1
params.a = 1
params.b = 0.01
params.M = 300
params.n_epochs = 100
params.max_iter = 1

data = load_cvae_data()
num_factors = 50
model = CVAE(num_users=27936, num_items=29526, num_factors=num_factors, params=params,
    input_dim=33348, dims=[200, 100], n_z=num_factors, activations=['sigmoid', 'sigmoid'],
    loss_type='cross-entropy', lr=0.001, random_seed=0, print_step=10, verbose=False)
model.load_model(weight_path="../model/pretrain")
model.run(data["train_users"], data["train_items"], data["test_users"], data["test_items"],
   data["content"], params)
model.save_model(weight_path="model-cf10/cvae-packagedata", pmf_path="model-cf10/pmf-packagedata")
#model.load_model(weight_path='model-cf10/cvae-packagedata', pmf_path='model-cf10/pmf-packagedata')
#model.evaluateCorrect(data['train_users'], data['test_users'], params.M)
# print("Recall of the model is {}".format(model.predict(data['train_users'], data['test_users'], params.M)))
