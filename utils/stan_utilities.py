import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
import pickle
import pystan


def get_model(model_code_file, model_pickle_file):
    """ Loads a STAN model into a pystan.model.StanModel object.

    If the the file atmodel_pickle_file exists, it loads it,
    but also checks if the code of the loaded model is 
    matches with the model code from model_code_file.
    If it does not match, the model is re-compiled, 
    and the pickle file is overwritten.

    Args:
        model_code_file (str): path to stan code file
        model_pickle_file (str): path to model pickle file

    Returns:
        pystan.model.StanModel: model object
    """

    try:
        f_model_code = open(model_code_file, 'r')
        model_code = f_model_code.read()
        f_model_code.close()
    except:
        raise 'Failed to read model_code_file {}.'.format(model_code_file)

    needs_rebuild = False
    try:
        f_model_pickle = open(model_pickle_file, 'rb')
        f_model_pickle.close()
    except:
        print('Failed to read model_pickle_file {}'.format(model_pickle_file))
        needs_rebuild = True

    if not needs_rebuild:
        with open(model_pickle_file, 'rb') as f_model_pickle:
            try:
                model = pickle.load(f_model_pickle)
                assert model.model_code.strip() == model_code.strip()
                return model
            except:
                print('Model in model_pickle_file {} is different '.format(model_pickle_file) + \
                      'from model in model_code_file {}.'.format(model_code_file))
                needs_rebuild = True

    if needs_rebuild:
        print('Compiling stan model ...')
        model = pystan.StanModel(model_code=model_code)
        with open(model_pickle_file, 'wb') as f_model_pickle:
            pickle.dump(model, f_model_pickle)
        print('Done.')
        return model


def plot_traces(fit, chains=None, include_warmup=False):
    """ Plot MC traces of all parameters of a StanFit4Model object.

    All chains and all parameters are plotted, each on a separate
    subplot, arranged in a table where rows are parameters and 
    columns are chains.

    Args:
        fit (StanFit4Model): result produced by pystan's sampling()
        chains (iterable): list of chain indexes to plot
        include_warmup (bool): if True, all samples are shown
    """

    total_chains = len(fit.sim['samples'])
    if chains is None:
        chains = list(range(total_chains))
    for chain in chains:
        assert 0 <= chain < total_chains
    labels = list(fit.sim['samples'][0]['chains'].keys())
    
    if include_warmup:
        t_start = 0
    else:
        t_start = fit.sim['warmup'] // fit.sim['thin']
    t_end = len(fit.sim['samples'][0]['chains'][labels[0]])
    t = list(range(t_start, t_end))
    
    fig, axes_rows = plt.subplots(len(labels), len(chains), 
                                  figsize=(3 * len(chains), 1 * len(labels)), 
                                  sharey='row')
    
    # plot traces
    for row_idx, label in enumerate(labels):
        axes = axes_rows[row_idx]
        label = labels[row_idx]
        for col_idx, chain in enumerate(chains):
            ax = axes[col_idx]
            ax.plot(t, fit.sim['samples'][chain]['chains'][label][t_start:t_end])
    
    # cosmetic changes to y axis labels
    for row_idx, label in enumerate(labels):
        axes = axes_rows[row_idx]
        ax = axes[0]
        ax.set_ylabel(label + ' '*20, rotation=0)
        for col_idx in range(1, len(chains)):
            ax = axes[col_idx]
            ax.yaxis.set_ticks_position('none')
            
    # cosmetic changes to x axis labels
    for row_idx, label in enumerate(labels[:-1]):
        axes = axes_rows[row_idx]
        for ax in axes:
            ax.xaxis.set_ticks_position('none')
            ax.set_xticklabels([])
    for col_idx, chain in enumerate(chains):
        ax = axes_rows[-1][col_idx]
        ax.set_xlabel('chain {}'.format(chain))
    
    
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    

def plot_posteriors(fit, chains=None, markers=None):
    """ Plots histograms of posteriors of all parameters.

    It produces stacked histograms, each chain denoted with a
    different color. Plus an overall kde estimate.

    The optional `markers` argument accepts a dictionary, mapping
    from parameter names to floats, which adds one red vertical 
    marker to the corresponding plots. This can be used to show the
    true value (in case of validation) or results other estimates,
    e.g. MLE.

    Args:
        fit (StanFit4Model): result produced by pystan's sampling()
        chains (iterable): list of chain indexes to include
        markers (dict): parameter name -> float map
    """

    total_chains = len(fit.sim['samples'])
    if chains is None:
        chains = list(range(total_chains))
    for chain in chains:
        assert 0 <= chain < total_chains
    labels = list(fit.sim['samples'][0]['chains'].keys())
    
    
    t_start = fit.sim['warmup'] // fit.sim['thin']
    t_end = len(fit.sim['samples'][0]['chains'][labels[0]])
    
    fig, axes = plt.subplots(len(labels), 1, 
                             figsize=(5, len(labels)*2))
    for idx, label in enumerate(labels):
        ax = axes[idx]
        samples = [fit.sim['samples'][chain]['chains'][label][t_start:t_end] for chain in chains]
        _, bins, _ = ax.hist(samples, bins=50, stacked=True, density=True, alpha=0.6)
        
        y = np.zeros_like(bins)
        for idx, chain in enumerate(chains):
            kde = gaussian_kde(samples[idx])
            y += kde.evaluate(bins) / float(len(chains))
        ax.plot(bins, y, color='black', lw=1)
            
        if markers is not None:
            if label in markers:
                ax.axvline(markers[label], color='red')
        
        ax.set_ylabel(label + ' '*20, rotation=0)
        ax.set_yticklabels([])
        ax.yaxis.set_ticks_position('none')
    
    fig.subplots_adjust(hspace=0.5)
    plt.show()
    

def posterior_stats(fit, chains=None):
    """ Calcualtes mean, std and correlation between parameters.

    Args:
        fit (StanFit4Model): result produced by pystan's sampling()
        chains (iterable): list of chain indexes to plot

    Returns:
        dict: with the following keys
            * 'par'  -> list of parameter names
            * 'mean' -> numpy.array, parameter means
            * 'std'  -> numpy.array, parameter standard deviations
            * 'corr' -> 2-d numpy.array, correlation matrix of the parameters
    """
    total_chains = len(fit.sim['samples'])
    if chains is None:
        chains = list(range(total_chains))
    for chain in chains:
        assert 0 <= chain < total_chains
    labels = list(fit.sim['samples'][0]['chains'].keys())
    
    t_start = fit.sim['warmup'] // fit.sim['thin']
    t_end = len(fit.sim['samples'][0]['chains'][labels[0]])
    
    par_chains = []
    for idx, label in enumerate(labels):
        samples = [fit.sim['samples'][chain]['chains'][label][t_start:t_end] for chain in chains]
        par_chains.append(np.array(samples).flatten())
    par_chains = np.array(par_chains)

    mean = np.mean(par_chains, axis=1)
    cov = np.cov(par_chains)
    std = np.sqrt(np.diag(cov))
    corr = np.corrcoef(par_chains)

    stats = {
        'par': labels,
        'mean': mean,
        'std': std,
        'corr': corr
    }
    return stats
