import tool
from parameters import *
import pripl_tree
import grids
import numpy as np


if __name__ == '__main__':

    # A example for 1-D range queries
    # 1st. read data
    data, domain_size, attributes = tool.load_data(arguments.dataset, dimension_num=1)
    # 2nd. build pripl tree
    priplt = pripl_tree.Pripl_tree(data, domain_size, epsilon = arguments.epsilon)
    built_tree = priplt.build_pripl_tree()
    # 3rd. read query
    queries = tool.load_queries(query_volume = arguments.query_volume, domain_sizes = domain_size, query_dimensions=1)
    a_query = queries[0]
    # 4th. respond to query
    est_distr = priplt.get_distribution_from_tree(built_tree)
    ans = np.sum(est_distr[a_query[0]:a_query[1]+1])
    print("the answer of Q({}) is {}".format(a_query, ans))

    # A example for ]ambda-D range queries
    # 1st. read data
    data, domain_sizes, attributes = tool.load_data(arguments.dataset, dimension_num=arguments.attribute_num)
    # 2nd. build pripl tree
    mix_structures = grids.Multi_grids(data, attributes, domain_sizes, arguments.epsilon)
    mix_structures.build_multi_grids()
    # 3rd. read query
    query_attr_indexes, query_ranges = tool.load_queries(query_volume = arguments.query_volume, domain_sizes = domain_sizes, query_dimensions=arguments.query_dimension)
    a_query_atrr = query_attr_indexes[0]
    a_query_range = query_ranges[0]
    # 4th. respond to query
    ans =  mix_structures.answer_range_query(a_query_range, a_query_atrr)
    print("the answer of Q({}) on attributes {} is {}".format(list(a_query_range), list(np.asarray(attributes)[a_query_atrr]), ans))
