import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors

from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.signal import savgol_filter

from qvi.core.experiments import compute_traces_from_multiple_trainning

class _TFColor(object):
    """Enum of colors used in TF docs."""
    red = '#F15854'
    blue = '#5DA5DA'
    orange = '#FAA43A'
    green = '#60BD68'
    pink = '#F17CB0'
    brown = '#B2912F'
    purple = '#B276B2'
    yellow = '#DECF3F'
    gray = '#4D4D4D'
    qmc = '#33a02c'
    rqmc = '#fdae61'
    qvi = '#2c7bb6'
    rqvi = '#abd9e9'
    mc = '#d7191c'

    def __getitem__(self, i):
        return [
            self.red,
            self.orange,
            self.green,
            self.blue,
            self.pink,
            self.brown,
            self.purple,
            self.yellow,
            self.gray,
        ][i % 9]


TFColor = _TFColor()


def plot2d(array, ax=None, **kwargs):
    ''' Two dimension plot of an array. 

    Args:
        array: 2D array. The right-most index is the dimension index.
        ax: Matplotlib Axes. If None, one would be created.
        kwargs: key word argument for the scatter plot.
    '''
    ax = plt.subplot() if ax is None else ax
    X = array[:, 0]
    Y = array[:, 1]
    ax.scatter(X, Y, **kwargs)


def plot_experiments(experiments, axes, dataset, limits=None, gradylimit=None, dataset_name=None, vi_type=['all'], abscissa='epochs', num_burnin_steps=0, norm=1, **kwargs):
    """Plotting MCVI, QVI and RQVI Experiment 
    Args:
      experiments: Iterable of Experiment instances.
      axes: Matplotlib Axe.
      dataset: Dataset Number. Must be comptatible with axes.
      limites: Python `tuple` for xlim and ylim for each plot in Axes.
      name: Python `str` name dataset name for title display.
      name: Iterable of Python `str`: 'All', 'mc', 'QVI', 'RQVI'.
      abscissa: Python `str`: `time` or `epochs`.
      num_burnin_steps: Python `int`: Number of step to ignore for display.
    """

    for k, experiment in enumerate(sorted(experiments, key=lambda e: e.optimizer_params['learning_rate'])):
        line = 2*dataset
        elboax = axes[line, k]
        gradax = axes[line+1, k]
        
        qmc_ls = 'x'
        rqmc_ls = 'v'
        qvi_ls = ''
        rqvi_ls = '.'
        mc_ls = '^'
        every = 70

        if 'all' in vi_type or 'rqvi' in vi_type:
            plot(elboax, gradax, experiment.rqvitraces, abscissa=abscissa,marker=rqvi_ls,markevery=every,
                 name='RQVI', log_scale=False, c=TFColor.rqvi, num_burnin_steps=num_burnin_steps, norm=norm,**kwargs)

        if 'all' in vi_type or 'qmc' in vi_type:
            plot(elboax, gradax, experiment.qmcvitraces, abscissa=abscissa,marker=qmc_ls,markevery=every,
                 name='QMCVI', log_scale=False, c=TFColor.qmc,num_burnin_steps=num_burnin_steps, gradaxalpha=.6, norm=norm,**kwargs)
            
        if 'all' in vi_type or 'rqmc' in vi_type:
            plot_multiple_traces(elboax, gradax, experiment.rqmcvitraces, color=TFColor.rqmc,marker=rqmc_ls,markevery=every,
                                 abscissa=abscissa, name='RQMC multiple', log_scale=True, num_burnin_steps=num_burnin_steps, norm=norm,**kwargs)
    
        if 'all' in vi_type or 'qvi' in vi_type:
            plot(elboax, gradax, experiment.qvitraces, abscissa=abscissa,marker=qvi_ls,markevery=every,
                 name='QVI', log_scale=False, c=TFColor.qvi, num_burnin_steps=num_burnin_steps,gradaxalpha=.6, norm=norm,**kwargs)
            
        
        if 'all' in vi_type or 'mc' in vi_type:
            plot_multiple_traces(elboax, gradax, experiment.mcvitraces,color=TFColor.red,marker=mc_ls,markevery=every,
                                 abscissa=abscissa, name='MC VI Multiple', log_scale=True, num_burnin_steps=num_burnin_steps, norm=norm,**kwargs)
        
        
        elboax.set_xticks(ticks=[])
        if abscissa == 'time':
            xlabel = 'time(s)'
        elif abscissa == 'epochs':
            xlabel = 'iterations'
            xlabel = None
        else:
            xlabel = None
        gradax.set_xlabel(xlabel)
        elboax.set_yscale('symlog')
        gradax.set_yscale('symlog')

        if limits is not None:
            xlim, ylim = limits
            elboax.set_xlim(xlim)
            gradax.set_xlim(xlim)
            elboax.set_ylim(ylim)
        if gradylimit is not None:
            xlim, _ = limits
            gradax.set_xlim(xlim)
            gradax.set_ylim(*gradylimit)
        if k != 0:
            gradax.set_yticks(ticks=[])
            elboax.set_yticks(ticks=[])

            gradax.tick_params(axis=u'y', which=u'both', length=0)
            elboax.tick_params(axis=u'both', which=u'both', length=0)

        if k == 0:
            elboax.set_ylabel('ELBO')
            gradax.set_ylabel(r'$\mathbb{E}|g|_{2}$')

        elboax.set_title('{}'.format(
            dataset_name) + r'$(\alpha=${:.0e})'.format(experiment.optimizer_params['learning_rate']))
        
def plot(elboax, gradax, trace, abscissa='time', name='', log_scale=False, c=None, num_burnin_steps=0, alpha=1, gradaxalpha=1,norm=1, **kwargs):
    loss, timestamps, grads = trace
    if abscissa == 'time':
        x = timestamps - timestamps[num_burnin_steps]
    elif abscissa == 'epochs':
        x = np.arange(0, len(loss))

    grads = tf.reduce_sum(tf.square(grads), axis=-1)/norm

    if log_scale is True:
        grads = tf.math.log(grads)
        loss = tf.math.log(loss)

    elboax.plot(x[num_burnin_steps:], -
                loss[num_burnin_steps:], c=c, label=name, alpha=alpha,**kwargs)
    gradax.plot(x[num_burnin_steps:],
                grads[num_burnin_steps:], c=c, label=name, alpha=gradaxalpha,**kwargs)
    
def plot_multiple_traces(elboax, gradax, traces, abscissa='time', name='', log_scale=False, num_burnin_steps=0,color='red', norm=1,**kwargs):
    losses, timestamps, grads = compute_traces_from_multiple_trainning(traces)
    if abscissa == 'time':
        x = timestamps - timestamps[num_burnin_steps]
    elif abscissa == 'epochs':
        x = np.arange(0, len(losses))

    mean_var_ts(ts=losses[num_burnin_steps:], x=x[num_burnin_steps:],
                ax=elboax, label='MCVI', log_scale=log_scale, coeff=-1, color=color, norm=1,**kwargs)
    mean_var_ts(ts=grads[num_burnin_steps:], x=x[num_burnin_steps:],
                ax=gradax, label='MCVI', log_scale=log_scale,color=color,norm=norm,**kwargs)

def mean_var_ts(ts, x, ax, label=None, log_scale=False,color='red', coeff=1, norm=1, lw=.7,**kwargs):
    window_length_mean = 51
    window_length_var = 51
    mean = tf.reduce_mean(ts/norm, axis=-1)
    variance = tf.math.reduce_std(ts/norm, axis=-1)
    smoothed_mean = savgol_filter(mean, window_length_mean, 2)
    smoothed_variance = savgol_filter(variance, window_length_var, 3)

    import pdb
    #pdb.set_trace()
    edgecolor = '#CC4F1B'
    alpha = .6
    if log_scale is True:
        logmean = tf.math.log(smoothed_mean)
        logvariance = smoothed_variance/smoothed_mean
        ax.fill_between(x, coeff*tf.math.exp(logmean-logvariance), coeff*tf.math.exp(logmean+logvariance),
                        alpha=alpha, edgecolor=edgecolor, facecolor=color)
        ax.plot(x, coeff*smoothed_mean, c=color, label=label, lw=lw,**kwargs)
    else:
        ax.plot(x, coeff*smoothed_mean, c=color, label=label, lw=lw,**kwargs)
        ax.fill_between(x, coeff*smoothed_mean-smoothed_variance, coeff*smoothed_mean+smoothed_variance,
                        alpha=alpha, edgecolor=edgecolor, facecolor=color)

        

        
def scatter_plot_voronoi(qdist,n, ax, title=''):  
    q_samples = qdist.sample(n)
    speed = qdist.weights
    minima = min(speed)
    maxima = max(speed)
    norm = colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.Reds)

    vor = Voronoi(q_samples.numpy())
    voronoi_plot_2d(vor, show_points=False, show_vertices=False, s=1,ax=ax)
    for r in range(len(vor.point_region)):
        if r == 12: 
            pass
        else:
            region = vor.regions[vor.point_region[r]]
            if not -1 in region:
                polygon = [vor.vertices[i] for i in region]
                plt.fill(*zip(*polygon), color=mapper.to_rgba(speed[r]))
    ax.plot(q_samples[:,0], q_samples[:,1], 'ko', markersize=2)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.title.set_text(title)

    
def plot_heatmap_2d(dist, xmin=-4.0, xmax=4.0, ymin=-4.0, ymax=4.0, mesh_count=1000, name=None):
    plt.figure()
    
    x = tf.linspace(xmin, xmax, mesh_count)
    y = tf.linspace(ymin, ymax, mesh_count)
    X, Y = tf.meshgrid(x, y)
    
    concatenated_mesh_coordinates = tf.transpose(tf.stack([tf.reshape(Y, [-1]), tf.reshape(X, [-1])]))
    prob = dist.prob(concatenated_mesh_coordinates)
    #plt.hexbin(concatenated_mesh_coordinates[:,0], concatenated_mesh_coordinates[:,1], C=prob, cmap='rainbow')
    prob = prob.numpy()
    
    plt.imshow(tf.transpose(tf.reshape(prob, (mesh_count, mesh_count))), origin="lower")
    plt.xticks([0, mesh_count * 0.25, mesh_count * 0.5, mesh_count * 0.75, mesh_count], [xmin, xmin/2, 0, xmax/2, xmax])
    plt.yticks([0, mesh_count * 0.25, mesh_count * 0.5, mesh_count * 0.75, mesh_count], [ymin, ymin/2, 0, ymax/2, ymax])
    if name:
        plt.savefig(name + ".png", format="png")
