import math
import pandas as pd
import numpy as np
import sys

class LM:
	
	def __init__(self, y, x):
		if not isinstance(x, pd.Series) and not isinstance(y, pd.Series):
			print('Error: Must pass a pandas series')
			sys.exit()
		if x.shape[0] != y.shape[0]:
			print('Error: X and Y must have the same number of observations')
			sys.exit()
		
		self.y = y
		self.x = x
		self.b1 = self.compute_b1()
		self.b0 = self.compute_b0()

	def compute_b1(self):
		numerator = cov(self.x, self.y)
		denominator = cov(self.x, self.x)
		return numerator / denominator

	def compute_b0(self):
		return self.y.mean() - self.b1 * self.x.mean()

def cov(x,y):
	if not isinstance(x, pd.Series) and not isinstance(y, pd.Series):
		print('Error: Must pass a pandas series')
		sys.exit()
	if x.shape[0] != y.shape[0]:
		print('Error: X and Y must have the same number of observations')
		sys.exit()
	
	df = pd.concat([x, y], axis = 1)
	df.columns = ['x', 'y']
	df['x_bar'] = df['x'].mean()
	df['y_bar'] = df['y'].mean()

	df['cov'] = (df['x'] - df['x_bar']) * (df['y'] - df['y_bar'])
	
	return df['cov'].sum() / (df.shape[0] - 1)

def cor(x, y):
	if not isinstance(x, pd.Series) and not isinstance(y, pd.Series):
		print('Error: Must pass a pandas series')
		sys.exit()
	if x.shape[0] != y.shape[0]:
		print('Error: X and Y must have the same number of observations')
		sys.exit()
	
	df = pd.concat([x, y], axis = 1)
	df.columns = ['x', 'y']
	
	# Do it the hard way
	s_x = std(df['x'])
	s_y = std(df['y'])
	
	return cov(df['x'], df['y']) / (s_x * s_y)

def std(x):
	if not isinstance(x, pd.Series):
		print('Error: Must pass a pandas series')
		sys.exit()
	
	x_bar = x.mean()
	x_bar_s = pd.Series(x_bar, index = [k for k in range(x.shape[0])])
	df = pd.concat([x, x_bar_s], axis = 1)
	df.columns = ['x', 'x_bar']

	df['z'] = (df['x'] - df['x_bar']) ** 2
	s_x_2 = df['z'].sum() / (df.shape[0] - 1)
	
	return math.sqrt(s_x_2)









