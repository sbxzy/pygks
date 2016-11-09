from pygks.utils import csv_reader
from pygks.ui_isoinn import data_block
from numpy import array

r = csv_reader('wine_train.csv')
data = r.get_all()
nn_model = data_block(data, age_max = 200, nn_lambda = 70)
nn_model.draw_2d()
