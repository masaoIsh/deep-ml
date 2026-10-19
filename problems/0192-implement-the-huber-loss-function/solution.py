import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
	"""
	Compute the Huber Loss between true and predicted values.

	Args:
		y_true (float | list[float]): Ground truth values
		y_pred (float | list[float]): Predicted values
		delta (float): Transition threshold between MSE and MAE behavior

	Returns:
		float: Average Huber loss
	"""
	# Your code here
	if np.isscalar(y_true):
		if np.abs(y_true-y_pred) <= delta:
			return 0.5*(y_true-y_pred)**2
		else:
			return delta*(np.abs(y_true-y_pred))-0.5*delta
		
	else:
		y_true = np.array(y_true, dtype=float)
		y_pred = np.array(y_pred, dtype=float)

		huber_losses = np.zeros_like(y_true)
		for i in range(len(huber_losses)):
			if np.abs(y_true[i] - y_pred[i]) <= delta:
				huber_losses[i] = 0.5 * (y_true[i]-y_pred[i])**2
			else:
				huber_losses[i] = delta*(np.abs(y_true[i]-y_pred[i]))-0.5*delta**2
		
		return np.mean(huber_losses)

