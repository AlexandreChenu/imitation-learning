from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import torch

sns.set(style='white')


# Cycles over an iterable without caching the order (unlike itertools.cycle)
def cycle(iterable):
  while True:
    for x in iterable:
      yield x


# Flattens a list of dicts with torch Tensors
def flatten_list_dicts(list_dicts):
  return {k: torch.cat([d[k] for d in list_dicts], dim=0) for k in list_dicts[-1].keys()}


# Makes a lineplot with scalar x and statistics of vector y
def lineplot(x, y_, y2=None, filename='', xaxis='Steps', yaxis='Return', title=''):
      
  y = np.array(y_).reshape(-1,1)
  y_mean, y_std = y.mean(axis=1).reshape(y.shape[0],), y.std(axis=1).reshape(y.shape[0],)
  sns.lineplot(x=x, y=y_mean, color='coral')
  plt.fill_between(x, y_mean - y_std, y_mean + y_std, color='coral', alpha=0.3)
  if y2:
    y2 = np.array(y2)
    y2_mean, y2_std = y2.mean(axis=1), y2.std(axis=1)
    sns.lineplot(x=x, y=y2_mean, color='b')
    plt.fill_between(x, y2_mean - y2_std, y2_mean + y2_std, color='b', alpha=0.3)

  plt.xlim(left=0, right=x[-1])
  plt.xlabel(xaxis)
  plt.ylabel(yaxis)
  plt.title(title)
  plt.savefig(f'{filename}.png')
  plt.close()


def plot_traj(env, trajs, traj_eval, save_dir, it=0):
	fig, ax = plt.subplots()

	env.plot(ax)

	# ax.set_xlim((-0.1, 4.))
	# ax.set_ylim((-0.1, 1.1))

	for traj in trajs:
		for i in range(traj[0].shape[0]):
			X = [state[i][0] for state in traj]
			Y = [state[i][1] for state in traj]
			Theta = [state[i][2] for state in traj]
			ax.plot(X,Y, marker=".", c="blue", alpha = 0.7)

			for x, y, t in zip(X,Y,Theta):
				dx = np.cos(t)
				dy = np.sin(t)
				#arrow = plt.arrow(x,y,dx*0.1,dy*0.1,alpha = 0.6,width = 0.01, zorder=6)

	X_eval = [state[0] for state in traj_eval]
	Y_eval = [state[1] for state in traj_eval]
	ax.plot(X_eval, Y_eval, c = "red")

	plt.savefig(save_dir + "trajs_it_"+str(it)+".png")
	plt.close(fig)
	return
