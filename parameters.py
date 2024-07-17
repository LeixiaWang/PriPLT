import argparse

def algo_args(): 
    parser = argparse.ArgumentParser(description='the arguments for our algorithm')

    # int type
    parser.add_argument("--user_num", type=int, default = 10 ** 6, help="the number of users")
    parser.add_argument("--attribute_num", type=int, default = 5, help="the number of attributes")
    parser.add_argument("--od_domain_size", type=int, default = 1024, help="the domain size of each attribute")
    parser.add_argument("--md_domain_size", type=int, default = 256, help="the domain size of each attribute")
    parser.add_argument("--query_dimension", type=int, default = 2, help= "the query dimension")
    parser.add_argument("--data_dimension", type=int, default=5, help='the dimension of dimensional data.')
    parser.add_argument("--search_granularity", type=int, default=2 ** 7, help='the granularity of search during PL fitting.')

    # float type
    parser.add_argument("--epsilon", type=float, default=0.8, help= "the privacy budget")
    parser.add_argument("--query_volume", type=float, default=0.5, help='the ratio of the query range to the domain size (for 1-d).')
    parser.add_argument("--alpha", type=float ,default=0.2, help="the ratio of users for piecewise linear regression")
    parser.add_argument("--beta", type=float, default=0.04, help="the paramenter in adaptive grids, i.e. the eta in the manuscript.")

    # str type
    parser.add_argument("--data_path", type=str, default='./datasets/', help='the path of dataset')
    parser.add_argument("--query_path", type=str, default='./query/', help="the path of queries")
    parser.add_argument("--dataset", type=str, default='loan', help="the default dataset")
    
    # 解析参数对象
    args, unknown = parser.parse_known_args()
    return args

arguments = algo_args()