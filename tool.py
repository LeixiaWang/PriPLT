import h5py
import numpy as np
from parameters import *

def load_data(data_name, dimension_num = 1, user_num = None, domain_size = None):
    '''
    load the specified data.

    Args:
        data_name(str):
        dimension_num(int): the default is 1-d. when it is specified as 0, it return all dimensions\
            , otherwise return the data with specified dimensional number.
        user_num(int): the default is -1, meaning all users. it can be specified to a value smaller than user number of the dataset
        domain_size(int): only synthetic data can specify the domain size.
    Returns:
        data(list): a 2-d np.ndarray
        domain_sizes(list): the domain_size of each dimension
        attributes(list): the attributes name of each dimension
        (correlation(float)): the covariance of multi-variate distribution for synthetic datasets
    '''
    # print('loading {} dataset ...'.format(data_name))
    path = '{}{}.hdf5'.format(arguments.data_path, data_name)
    with h5py.File(path, 'r') as rf:
        data = rf['data']
        # data is a h5py type object, has the following attributes:
        # - data.shape: shape
        # - data.attrs['domain_size']: the domain size of each attribute, which is a list for real datasets, but a value for synthetic datasets
        # - data.attrs['correlation']: the covariance between data of different attibutes in synthetic datasets
        # - list(data.attrs['description']): the name of data attributes
        # - data.attrs['default_attribute']: the default attribute for 1-d range query
        # - data[:,index]: get the specified column of given indexes
        # print("the number of attributes:", len(data.attrs['description']))
        rng = np.random.default_rng(0)
        if dimension_num == 0: # return all attributes
            index = range(len(data.attrs['description']))
        else: # return attributes randomly
            index = range(dimension_num)
        # select dimension
        selected_data = data[:,index]
        rng.shuffle(selected_data) # shuffle all users
        attributes = list(data.attrs['description'][index])
        # select users
        if user_num is not None and user_num < len(selected_data):
            selected_data = selected_data[:user_num]
        # given domain
        domain_sizes = data.attrs['domain_size'][:dimension_num] if dimension_num > 0 else data.attrs['domain_size']
        if domain_size is not None:
            domain_size = np.array([min(domain_size, domain_sizes[i]) for i in range(len(domain_sizes))])
            selected_data = (selected_data / (domain_sizes /domain_size)).astype(int)
            domain_sizes = domain_size
        domain_sizes = domain_sizes[0] if dimension_num == 1 else domain_sizes 
        # extract the correlation of synthetic data
        corre_exist = False
        if 'correlation' in list(data.attrs.keys()):
            corre_exist = True
            correlation = data.attrs['correlation']
    if corre_exist:
        res = (selected_data, domain_sizes, attributes, correlation)
    else:
        res = (selected_data, domain_sizes, attributes)
    # print("successfully loaded.")
    return res

def load_queries(query_volume = arguments.query_volume, domain_sizes = arguments.od_domain_size, query_dimensions = arguments.query_dimension):
    '''
    Args:
        query_volme(float): the ratio of the query range to the domain size
        domain_size(int): the domain sizes of each dimension
            for 1-d cases, the domain size is a value
            for mult-d cases, the doamin sizes is a list or a np.ndarray, storing multiple domain sizes of attributes
        query_dimensions(int): the dimension of the query
    '''
    if np.isscalar(domain_sizes) and query_dimensions > 1:
        raise Exception('Error in loading queries: the domain sizes do not match the dimension.')
    file_name = arguments.query_path + 'queries_{}.hdf5'.format(query_volume)
    with h5py.File(file_name, 'r') as rf:
        original_queries = rf['queries'][:]
        query_num = rf['queries'].attrs['query_number']
    if np.isscalar(domain_sizes):
        rng = np.random.default_rng(0)
        queries_td = original_queries.copy()
        rng.shuffle(queries_td)
        queries = np.floor(np.array(queries_td) * domain_sizes).astype(int).copy()
        # print('1-d queries is loaded.')
        return queries.tolist()
    else:
        rng = np.random.default_rng(0)
        query_attrs = np.empty((query_num, query_dimensions), dtype=int)
        for j in range(query_num):
            query_attrs[j] = rng.choice(len(domain_sizes), query_dimensions, replace=False)
        query_domains = domain_sizes[query_attrs]
        queries = np.empty((query_num, query_dimensions), dtype=object)
        for i in range(query_dimensions):
            rng = np.random.default_rng(i)
            queries_td = original_queries.copy()
            rng.shuffle(queries_td)
            queries_1d = np.floor(np.array(queries_td) * np.tile(query_domains[:,i].reshape(query_num, 1),2)).astype(int).copy()
            queries[:, i] = queries_1d.tolist()
        # print('Multi-d queries is loaded.')
        return query_attrs, np.array(queries.tolist())

def slices_sum(data, sub_range):
    data = np.append(data, 0)
    ind = sub_range.T.ravel(order='F')
    if ind[-1] == data.size:
        ind = ind[:-1]
    return np.add.reduceat(data.ravel(), ind)[::2]

def mre(est:np.array, real:np.array):
    '''
    mean relative value: give up this matrics since the real answer could be 0
    '''
    absolute_error = np.abs(est -  real)
    mres = absolute_error / real
    return np.average(mres)

def mse(values1:np.array, values2:np.array):
    '''
    mean square error
    '''
    mses = np.square(values1 - values2)
    return np.average(mses)

def count(data, domain_sizes: int or tuple):
    if np.size(domain_sizes) == 1:
        hist = np.zeros(domain_sizes)
        unique, counts = np.unique(data, return_counts=True)
        for i in range(len(unique)):
            hist[int(unique[i])] = counts[i]
        return hist / len(data)
    else:
        unique, counts = np.unique(data, axis=0, return_counts=True)
        hist = np.zeros((len(unique), len(domain_sizes) + 1))
        hist[:, :-1] = unique
        hist[:, -1] = counts / len(data)
        return hist
    
def query_once_from_multi_dims(real_hist, query_attr_indx, query_ranges):
    cond = np.ones(len(real_hist), dtype=bool)
    for i in range(len(query_attr_indx)):
        cond &= (real_hist[:,query_attr_indx[i]] >= query_ranges[i][0]) & (real_hist[:,query_attr_indx[i]] <= query_ranges[i][1])
    return np.sum(real_hist[:,-1][cond])
