import numpy as np
from parameters import *
from frequency_oracle import *
from pripl_tree import *
from scipy.special import comb
import itertools


class Grid(object):
    def __init__(self, dim:int, attribute_indexes:list, attribute_names:tuple, domains:tuple, user_data:np.array = None):
        self.dim = dim
        self.attribute_indexes = attribute_indexes
        self.attribute_names = attribute_names
        self.domain_sizes = domains
        self.data = user_data
        if dim == 2: # (grids_ids, node_ids)
            self.merging_relation = {self.attribute_indexes[0]:[], self.attribute_indexes[1]:[]} 
            self.splitting_relation = {self.attribute_indexes[0]:[], self.attribute_indexes[1]:[]} 
            self.corresponding_relation = {self.attribute_indexes[0]:[], self.attribute_indexes[1]:[]}
    
    def set_grid(self, partitions = None, frequencies = None, var = None):
        if partitions is not None:
            self.grid_partitions = partitions
            # a 2-d array: grid_partitions[0] is the partition of the x axis; grid_partitions[1] is the partition of the y axis.
        if frequencies is not None:
            self.grid_frequencies = frequencies
            # a 2-d array: with the size of len(grid_paritions[0]) * len(grid_partitions[1])
        if var is not None:
            self.grid_var = var
            # a 2-d array: with the size of len(grid_paritions[0]) * len(grid_partitions[1])

    def set_distribution(self, distribution):
        self.distribution = distribution  

    def add_node_mapping(self, attr_index, relation:{'splitting', 'merging'}, node_ids: list | int, grid_ids: list | int):
        if relation == 'splitting':
            assert isinstance(node_ids, int) and isinstance(grid_ids, list)
            self.splitting_relation[int(attr_index)].append((grid_ids, node_ids))
        if relation == 'merging':
            assert isinstance(grid_ids, int) and isinstance(node_ids, list)
            self.merging_relation[int(attr_index)].append((grid_ids, node_ids))
        if relation == 'corresponding':
            assert isinstance(grid_ids, int) and isinstance(node_ids, int)
            self.corresponding_relation[int(attr_index)].append((grid_ids, node_ids))


class Multi_grids(object):
    '''
    Utilizing the pripl-tree, building the 2-d grids, answering the multi-dimensional range query
    '''
    def __init__(self, data, attributes, domain_sizes, epsilon = arguments.epsilon, beta = arguments.beta):
        self.data = data
        self.attributes = np.array(attributes)
        self.domain_sizes = np.array(domain_sizes) # one-to-one correspondence to the attribtues
        self.epsilon = epsilon
        self.user_num = len(data)
        self.sigma_square = 4 * math.exp(self.epsilon) / (self.user_num * (math.exp(self.epsilon - 1)**2))
        self.tree_set,self.grid_set = {}, {}
        self.beta = beta
        self.iterative_maximal_number = 1000
        self.update_error = True


    def build_multi_grids(self, run_time = False):
        # estimate 1-d distribution with PriPL-Tree
        self._1d_estimation()
        # partition these grids adaptively
        self._adaptive_grid_partition()
        # collect frequency for each cell of grids
        self._2d_grids_estiamtion()
        # constrained inference and update the trees and the grids
        self._constrained_inference()
        # build 2-d distribution metrics
        self._distirbution_estimation() 
    
    
    def _1d_estimation(self):
        # allocate a half of users for 1-D tree estimation
        m = len(self.attributes)
        user_num = int(self.user_num / 2 / m)
        end_id = int(self.user_num / 2)
        for i in range(m):
            user_start_id = i * user_num
            user_end_id = (i+1) * user_num if i < m-1 else end_id
            data = self.data[user_start_id: user_end_id]
            pripl_tree = Pripl_tree(data[:, i], self.domain_sizes[i], self.epsilon)
            # estimate the pripl tree for each dimension
            pripl_tree.build_pripl_tree()
            self.tree_set[i] = pripl_tree
            # store in 1-D grids
            grid = Grid(1, i, self.attributes[i], self.domain_sizes[i], data[:, i])
            grid.set_distribution(pripl_tree.get_distribution_from_tree())
            leaves = pripl_tree.get_leaves_in_order()
            partitions = np.zeros((len(leaves),2), dtype = int)
            frequencies = np.zeros(len(leaves))
            var = np.zeros(len(leaves))
            for j, leaf in enumerate(leaves):
                partitions[j] = [leaf.data.interval_left, leaf.data.interval_right]
                frequencies[j] = leaf.data.frequency
                var[j] = leaf.data.error
            grid.set_grid(partitions, frequencies, var)
            self.grid_set[i] = grid
            

    def _adaptive_grid_partition(self):
        pairs = np.array(list(itertools.combinations(range(len(self.attributes)), 2))) # a 2-d array
        t = int(comb(len(self.attributes), 2))
        # initialize 2-d grids
        user_num = int(self.user_num / 2 / t)
        start_id = int(self.user_num / 2)
        for i, pair in enumerate(pairs):
            user_start_id = start_id + i * user_num
            user_end_id = start_id + (i+1) * user_num if i < t-1 else self.user_num
            data = self.data[user_start_id: user_end_id]
            grid = Grid(2, pair, self.attributes[pair], self.domain_sizes[pair], data[:, pair])
            self.grid_set[tuple(pair)] = grid
        del self.data # this command aims to free the memory of data
        # adaptively grid partitioning
        est_std = math.sqrt(2 * t * self.sigma_square)
        for grid_id, grid in self.grid_set.items():
            if not isinstance(grid_id, tuple):
                continue
            # initialize some temprary variable of x and y axises of the grid
            x_id = grid_id[0]
            x_partitions = self.grid_set[x_id].grid_partitions.copy()
            x_grid_freq = self.grid_set[x_id].grid_frequencies.copy()
            x_distribution = self.grid_set[x_id].distribution.copy()
            y_id = grid_id[1]
            y_partitions = self.grid_set[y_id].grid_partitions.copy()
            y_grid_freq = self.grid_set[y_id].grid_frequencies.copy()
            y_distribution = self.grid_set[y_id].distribution.copy()
            # relation_record: store the origianl id of 1-D cells
            # - positive list: the corresponding ids during merging
            # - str value: the corresponding ids during splitting
            x_relation_record = np.empty(len(x_partitions), dtype=object)
            for i in range(len(x_partitions)):
                x_relation_record[i] = i
            y_relation_record = np.empty(len(y_partitions), dtype=object)
            for i in range(len(y_partitions)):
                y_relation_record[i] = i
            # status: indicating whether a cell can be splitted or merged further
            # - postive: 1 -> original status, to be splitted, can transform to {-1, 2, 0}
            #            2 -> splitted status, to be splitted, can transform to {2, 0}
            # - negative: -1 -> to be merged, can transforms to {-1, 0}
            # 0 : terminal status, can not split or merge anymore
            x_status = np.ones(len(x_partitions))
            y_status = np.ones(len(y_partitions))
            # first, we perform splitting
            while any(x_status > 0) or any(y_status > 0):
                x_candidate_index = np.where(x_status > 0)[0]
                x_max_id = x_candidate_index[np.argmax(x_grid_freq[x_candidate_index])] if any(x_status > 0) else None
                y_candidate_index = np.where(y_status > 0)[0]
                y_max_id = y_candidate_index[np.argmax(y_grid_freq[y_candidate_index])] if any(y_status > 0) else None
                # split x axis
                if (y_max_id is None) or ((x_max_id is not None) and (x_grid_freq[x_max_id] >= y_grid_freq[y_max_id])):
                    x_relation_record, x_status, x_partitions, x_grid_freq = self.__split_the_cell(t, x_max_id, x_partitions, x_grid_freq, x_distribution, y_partitions, est_std, x_relation_record, x_status)
                # splite y axis
                else:
                    y_relation_record, y_status, y_partitions, y_grid_freq = self.__split_the_cell(t, y_max_id, y_partitions, y_grid_freq, y_distribution, x_partitions, est_std, y_relation_record, y_status)
            # seconde, we perform merging
            while any(x_status < 0) or any(y_status < 0):
                x_candidate_index = np.where(x_status < 0)[0]
                x_min_id = x_candidate_index[np.argmin(x_grid_freq[x_candidate_index])] if any(x_status < 0) else None
                y_candidate_index = np.where(y_status < 0)[0]
                y_min_id = y_candidate_index[np.argmin(y_grid_freq[y_candidate_index])] if any(y_status < 0) else None
                # merge x axis
                if (y_min_id is None) or ((x_min_id is not None) and (x_grid_freq[x_min_id] <= y_grid_freq[y_min_id])): 
                    x_relation_record, x_status, x_partitions, x_grid_freq = self.__merge_the_cells(t, x_min_id, x_partitions, x_grid_freq, y_partitions, est_std, x_relation_record, x_status)
                # merge y axis
                else:
                    y_relation_record, y_status, y_partitions, y_grid_freq = self.__merge_the_cells(t, y_min_id, y_partitions, y_grid_freq, x_partitions, est_std, y_relation_record, y_status)
            # store the partitions information to the grid
            grid.set_grid(partitions = [x_partitions, y_partitions]) 
            # store the node relationship of splitting and merging
            grid = self.__store_cell_relations(x_id, grid, x_relation_record)
            grid = self.__store_cell_relations(y_id, grid, y_relation_record)             


    def __split_the_cell(self, t, x_max_id, x_partitions, x_grid_freq, x_distribution, y_partitions, est_std, x_relation_record, x_status):
        # 1st. the frequency to be split should be larger than standard variance
        if x_grid_freq[x_max_id] > est_std:
            # 2nd. find the best split point
            new_parts = self.__find_best_split_point(x_partitions[x_max_id], x_distribution, est_std)
            if new_parts is None:
                x_status[x_max_id] = -1 if x_status[x_max_id] == 1 else 0
                # in this step, we can choose to transform the status to 0 directly
                # as such, the merging step only rely the frequncy thresh
            else:
                # 3rd. decide wether to split
                f_l = np.sum(x_distribution[new_parts[0][0]: new_parts[0][1]+1])
                f_r = np.sum(x_distribution[new_parts[1][0]: new_parts[1][1]+1])
                area_ratio = (new_parts[0][1] - new_parts[0][0] + 1) / (x_partitions[x_max_id][1] - x_partitions[x_max_id][0] + 1)
                choice = self.__whether_one_or_multi(t, len(x_partitions), len(y_partitions), f_l, f_r, area_ratio)
                if choice == 2:
                    # 4th. add splited partitions and grid frequency, update the relation record and status
                    x_partitions[x_max_id] = new_parts[1]
                    x_partitions = np.insert(x_partitions, x_max_id, new_parts[0], axis = 0)
                    x_grid_freq[x_max_id] = f_r
                    x_grid_freq = np.insert(x_grid_freq, x_max_id, f_l, axis = 0)
                    x_relation_record[x_max_id] = str(x_relation_record[x_max_id])
                    x_relation_record = np.insert(x_relation_record, x_max_id, x_relation_record[x_max_id], axis = 0)
                    x_status[x_max_id] = 2
                    x_status = np.insert(x_status, x_max_id, 2, axis=0)
                else:
                    x_status[x_max_id] = -1 if x_status[x_max_id] == 1 else 0 
        else:
            x_status[x_max_id] = -1 if x_status[x_max_id] == 1 else 0
        return x_relation_record, x_status, x_partitions, x_grid_freq


    def __merge_the_cells(self, t, x_min_id, x_partitions, x_grid_freq, y_partitions, est_std, x_relation_record, x_status):
        # 1st. find the adjacennt neighor with a smaller frequency
        left_neighbor_freq = x_grid_freq[x_min_id - 1] if x_min_id > 0 and x_status[x_min_id - 1] == -1 else np.inf
        right_neighbor_freq = x_grid_freq[x_min_id + 1] if x_min_id < len(x_partitions) - 1 and x_status[x_min_id + 1] == -1 else np.inf
        if left_neighbor_freq == np.inf and right_neighbor_freq == np.inf:
            x_status[x_min_id] = 0
        else:
            neighbor_id = x_min_id - 1 if left_neighbor_freq <= right_neighbor_freq else x_min_id + 1
            left_id = min(neighbor_id, x_min_id)
            right_id = max(neighbor_id, x_min_id) 
            # 2nd. decide whether to merge
            if x_grid_freq[left_id] < est_std and x_grid_freq[right_id] < est_std:
                choice = 1
            else:
                area_ratio = (x_partitions[left_id][1] - x_partitions[left_id][0] + 1) / (x_partitions[right_id][1] - x_partitions[left_id][0] + 1)
                choice = self.__whether_one_or_multi(t, len(x_partitions)-1, len(y_partitions), x_grid_freq[left_id], x_grid_freq[right_id], area_ratio)
            # 3rd. update merged partitions, merged frequencies, relation record and status
            if choice == 1:
                x_partitions[left_id] = np.array([x_partitions[left_id][0], x_partitions[right_id][1]])
                x_partitions = np.delete(x_partitions, right_id, axis=0)
                x_grid_freq[left_id] = x_grid_freq[left_id] + x_grid_freq[right_id]
                x_grid_freq = np.delete(x_grid_freq, right_id, axis=0)
                left_relation_record = x_relation_record[left_id] if isinstance(x_relation_record[left_id], list) else [x_relation_record[left_id]]
                right_relation_record = x_relation_record[right_id] if isinstance(x_relation_record[right_id], list) else [x_relation_record[right_id]]
                x_relation_record[left_id] = left_relation_record + right_relation_record
                x_relation_record = np.delete(x_relation_record, right_id, axis=0)
                x_status = np.delete(x_status, right_id, axis=0)
            else:
                x_status[left_id] = 0
                x_status[right_id] = 0
        return x_relation_record, x_status, x_partitions, x_grid_freq


    def __find_best_split_point(self, seg, whole_frequencies, y_grid_freq, thresh):
        min_loss = np.inf
        min_indx = -1
        for i in range(seg[0], seg[1]):
            f_l = np.sum(whole_frequencies[seg[0]:i+1])
            f_r = np.sum(whole_frequencies[i+1:seg[1]+1])
            if f_l < thresh or f_r < thresh:
                continue
            else:
                loss = np.sum(np.square(f_l * y_grid_freq)) + np.sum(np.square(f_r * y_grid_freq))
                if min_loss > loss:
                    min_loss = loss
                    min_indx = i
        if min_indx == -1:
            return None
        else:
            return [[seg[0], min_indx], [min_indx+1, seg[1]]]


    def __whether_one_or_multi(self, t, g_decide, g_given, f_l, f_r, y_grid_freq):
        one_grid_error = 2 * t * self.sigma_square * g_decide * g_given + self.beta * np.sum(np.square((f_l + f_r)*y_grid_freq))
        multi_grid_error = 2 * t * self.sigma_square * (g_decide + 1) * g_given + self.beta * (np.sum(np.square(f_l * y_grid_freq)) + np.sum(np.square(f_r * y_grid_freq)))
        return 1 if one_grid_error <= multi_grid_error else 2


    def __store_cell_relations(self, x_id, grid, x_relation_record):
        splitted_node = -1
        splitted_cells = []
        for cell_id, node_ids in enumerate(x_relation_record):
            if isinstance(node_ids, list):
                grid.add_node_mapping(x_id, 'merging', node_ids, cell_id)
            elif isinstance(node_ids, str):
                if splitted_node == -1:
                    splitted_node = int(node_ids)
                    splitted_cells.append(cell_id)
                elif splitted_node == int(node_ids):
                    splitted_cells.append(cell_id)
                else:
                    grid.add_node_mapping(x_id, 'splitting', splitted_node, splitted_cells)
                    splitted_node = int(node_ids)
                    splitted_cells = [cell_id]
            else:
                grid.add_node_mapping(x_id, 'corresponding', node_ids, cell_id)
        if len(splitted_cells) > 0:
            grid.add_node_mapping(x_id, 'splitting', splitted_node, splitted_cells)
        return grid


    def _2d_grids_estiamtion(self):
        for grid_id, grid in self.grid_set.items():
            if not isinstance(grid_id, tuple):
                continue
            fo = Frequency_oracle(grid.data, frequency_oracle_name='OUE', epsilon=self.epsilon, domain_size=grid.domain_sizes, merged_domain=grid.grid_partitions)
            grid_vars = np.full((len(grid.grid_partitions[0]), len(grid.grid_partitions[1])), fo.get_one_theoretical_square_error())
            grid.set_grid(frequencies = fo.get_aggregated_frequency(), var = grid_vars)


    def _constrained_inference(self):
        # update 1-d grids with minimal variance
        for grid_id, grid in self.grid_set.items():
            if isinstance(grid_id, tuple):
                for relation_dict in [grid.corresponding_relation, grid.merging_relation, grid.splitting_relation]:
                    for attr_id, relation in relation_dict.items():
                        for cell_ids, node_ids in relation:
                            self.__update_local_minimal_frequency(grid, attr_id, cell_ids, node_ids)
        for gird_id, grid in self.grid_set.items():
            if isinstance(gird_id, int):
                grid.grid_frequencies = self.__weighted_averaging(grid.grid_frequencies, 1, non_negative=True)
                self.tree_set[gird_id].refine_pripl_tree_leaves(grid.grid_frequencies)
        # update 2-d grids with minimal variance
        for grid_id, grid in self.grid_set.items():
            if isinstance(grid_id, tuple):
                old_frequencies = np.zeros(np.shape(grid.grid_frequencies))
                while np.sum(np.abs(grid.grid_frequencies - old_frequencies)) > 1 / (self.user_num ** 2):
                    old_frequencies = grid.grid_frequencies.copy()
                    for relation_dict in [grid.corresponding_relation, grid.merging_relation, grid.splitting_relation]:
                        for attr_id, relation in relation_dict.items():
                            for cell_ids, node_ids in relation:
                                self.__update_local_frequency_given_optima(grid, attr_id, cell_ids, node_ids)
                grid.grid_frequencies = self.__weighted_averaging(grid.grid_frequencies, 1, non_negative=True)   


    def __update_local_minimal_frequency(self, grid, attr_id, grid_ids, node_ids):
        '''
        only update 1-d grids
        '''
        # label the corresponding cells and nodes
        cell_mask = np.zeros(np.shape(grid.grid_frequencies), dtype=bool)
        if attr_id == grid.attribute_indexes[0]:
            cell_mask[grid_ids, :] = True
        else:
            cell_mask[:, grid_ids] = True
        # compute the optimal frequency
        if isinstance(node_ids, int):
            freqs_to_comp = [grid.grid_frequencies[cell_mask].sum(), self.grid_set[attr_id].grid_frequencies[node_ids]]
            vars_to_comp = [grid.grid_var[cell_mask].sum(), self.grid_set[attr_id].grid_var[node_ids].sum()]
            optimal_frequency, optimal_var = self.__minimize_variance(freqs_to_comp, vars_to_comp)
            # update the node
            self.grid_set[attr_id].grid_frequencies[node_ids] = optimal_frequency
            self.grid_set[attr_id].grid_var[node_ids] = optimal_var
        else:
            freqs_to_comp = [grid.grid_frequencies[cell_mask].sum(), self.grid_set[attr_id].grid_frequencies[node_ids].sum()]
            vars_to_comp = [grid.grid_var[cell_mask].sum(), self.grid_set[attr_id].grid_var.sum()]
            optimal_frequency, alpha = self.__minimize_variance(freqs_to_comp, vars_to_comp, True)
            # update the coefficients (step 1)
            error_vector = np.append(self.grid_set[attr_id].grid_var[node_ids], grid.grid_var[cell_mask].sum())
            error_coef_matrix = np.identity(len(error_vector))
            coef_1 = np.zeros(len(node_ids) + 1)
            coef_1[-1] = alpha[0]
            coef_1[:-1] = alpha[1]
            error_coef_matrix[-1] = np.dot(coef_1, error_coef_matrix)
            # update the node
            update_frequencies = self.__weighted_averaging(self.grid_set[attr_id].grid_frequencies[node_ids], optimal_frequency, non_negative = False) 
            self.grid_set[attr_id].grid_frequencies[node_ids]  = update_frequencies.copy()
            # update the coefficients (step 2)
            if self.update_error:
                coef_2 = np.identity(len(node_ids) + 1)[:len(node_ids)]
                mask = (update_frequencies > 0)
                if len(np.where(mask)[0]) > 0:
                    coef_2[mask,:-1] -= 1 / len(np.where(mask)[0])
                    coef_2[mask,-1] = 1 / len(np.where(mask)[0])
                coef = np.dot(coef_2, error_coef_matrix)
                self.grid_set[attr_id].grid_var[node_ids] = np.dot(np.square(coef), error_vector).reshape(-1)

    
    def __update_local_frequency_given_optima(self, grid, attr_id, grid_ids, node_ids):
        # label the corresponding cells and nodes
        cell_mask = np.zeros(np.shape(grid.grid_frequencies), dtype=bool)
        if attr_id == grid.attribute_indexes[0]:
            cell_mask[grid_ids, :] = True
        else:
            cell_mask[:, grid_ids] = True
        # extract the optimal frequency
        optimal_frequency = np.sum(self.grid_set[attr_id].grid_frequencies[node_ids])
        # update the cells
        if np.sum(cell_mask) == 1:
            grid.grid_frequencies[cell_mask] = optimal_frequency
        else:
            grid.grid_frequencies[cell_mask] = self.__weighted_averaging(grid.grid_frequencies[cell_mask], optimal_frequency, non_negative = True)


    def __minimize_variance(self, xs, vars, return_alpha = False):
        xs = np.array(xs)
        vars = np.array(vars)
        if any(vars == 0):
            index = np.where(vars == 0)[0]
            alpha = np.zeros(len(vars))
            alpha[index] = 1
        else:
            alpha = 1 / (vars * np.sum(1/vars, axis = 0))
        update_xs = np.sum(xs * alpha, axis = 0)
        udpate_vars = np.sum(vars * np.square(alpha), axis = 0)
        if return_alpha:
            return update_xs, alpha
        else:
            return update_xs, udpate_vars
    

    def __weighted_averaging(self, children_f:np.array, parent_f:float, non_negative = True):
        if parent_f == 0:
            C = np.zeros(np.shape(children_f))
        else:
            C = children_f.copy()
            C = C - (C.sum() - parent_f) / np.size(C)
            if non_negative:
                while (C < 0).any() and (C > 0).any():
                    C[C < 0] = 0
                    mask = (C > 0) 
                    if np.sum(mask) == 0:
                        print(parent_f,children_f)
                    C[mask] += (parent_f - C.sum()) / np.sum(mask)
        return C
    

    def _distirbution_estimation(self):
        for grid_id, grid in self.grid_set.items():
            # for 1-d grid
            if isinstance(grid_id, int):
                distribution = self.tree_set[grid_id].get_distribution_from_tree()
                grid.set_distribution(distribution)
            else:
                # for 2-d grid
                X = self.grid_set[grid_id[0]]
                Y = self.grid_set[grid_id[1]]
                distribution = np.full((grid.domain_sizes[0], grid.domain_sizes[1]), 1 / (grid.domain_sizes[0] * grid.domain_sizes[1]))
                old_distribution = np.zeros(distribution.shape)
                while True:
                    # 1st. constraints of X
                    x_sum = np.sum(distribution, axis=1)
                    update_x = np.where(x_sum != 0)[0]
                    distribution[update_x, :] = distribution[update_x, :] * np.expand_dims(X.distribution[update_x] / x_sum[update_x],1).repeat(np.shape(distribution)[1], axis = 1)
                    # 2nd. constraints of Y
                    y_sum = np.sum(distribution, axis=0)
                    update_y = np.where(y_sum != 0)[0]
                    distribution[:, update_y] = distribution[:, update_y] * np.expand_dims(Y.distribution[update_y] / y_sum[update_y],0).repeat(np.shape(distribution)[0], axis = 0)
                    # 3rd. constraints of grids
                    for j, x_seg in enumerate(grid.grid_partitions[0]):
                        for k, y_seg in enumerate(grid.grid_partitions[1]):
                            grid_distr = distribution[x_seg[0]:x_seg[1]+1, y_seg[0]:y_seg[1]+1]
                            if np.sum(grid_distr) != 0:
                                distribution[x_seg[0]:x_seg[1]+1, y_seg[0]:y_seg[1]+1] = grid_distr / np.sum(grid_distr) * grid.grid_frequencies[j,k]
                    distribution = distribution / np.sum(distribution)
                    difference = np.sum(np.abs(old_distribution - distribution))
                    old_distribution = distribution.copy()
                    if difference < 1 / (self.user_num ** 2):
                        break
                grid.set_distribution(distribution)


    def answer_range_query(self, query_ranges:list, attribute_indexes:list | int = None, attribute_names:list | str = None, run_time = False):
        # extract the indexes of attributes of queries
        if attribute_names is not None:
            attribute_indexes = np.where(self.attributes == np.array(attribute_names)[:,None])[-1] if not np.isscalar(attribute_names) else np.where(self.attributes == attribute_names)[0][0]
        elif attribute_indexes is None:
            attribute_indexes = np.random.choice(len(self.attributes), len(query_ranges), replace=False)
        # order the query ranges by the indexes
        if not np.isscalar(attribute_indexes) and len(attribute_indexes) > 1:
            query_ranges = query_ranges[np.argsort(attribute_indexes)]
            attribute_indexes = np.sort(attribute_indexes)
        else:
            attribute_indexes = int(attribute_indexes)
        # 1st. process 1-d query
        if isinstance(attribute_indexes, int):
            answer = self.grid_set[attribute_indexes].distribution[query_ranges[0][0]:query_ranges[0][1]+1].sum()
        # 2nd. process 2-d query
        elif len(attribute_indexes) == 2:
            if tuple(attribute_indexes) in self.grid_set:
                # using the estiamted 2-d grid
                answer = self.grid_set[tuple(attribute_indexes)].distribution[query_ranges[0][0]:query_ranges[0][1]+1, query_ranges[1][0]:query_ranges[1][1]+1].sum()
            else:
                # using the multiply result from 1-d distributions
                x = self.grid_set[attribute_indexes[0]].distribution[query_ranges[0][0]:query_ranges[0][1]+1].sum()
                y = self.grid_set[attribute_indexes[1]].distribution[query_ranges[1][0]:query_ranges[1][1]+1].sum()
                answer = x * y
        # 3rd. process multi-d query
        else:
            # find the associated queries
            constraints_1d = np.zeros((len(attribute_indexes),2))
            for i, index in enumerate(attribute_indexes):
                constraints_1d[i,1] = self.grid_set[index].distribution[query_ranges[i][0]:query_ranges[i][1]+1].sum()
                constraints_1d[i,0] = 1 - constraints_1d[i,1]
            constraints_2d = {}
            for index in filter(lambda x: x in self.grid_set,list(itertools.combinations(attribute_indexes, 2))):
                grid = self.grid_set[index]
                x = np.where(attribute_indexes == index[0])[0][0]
                y = np.where(attribute_indexes == index[1])[0][0]
                x_range = slice(query_ranges[x][0], query_ranges[x][1]+1)
                x_complement_l = slice(None, query_ranges[x][0])
                x_complement_r = slice(query_ranges[x][1]+1, None)
                y_range = slice(query_ranges[y][0], query_ranges[y][1]+1)
                y_complement_l = slice(None, query_ranges[y][0])
                y_complement_r = slice(query_ranges[y][1]+1, None)
                constraints_2d[index] = {}
                constraints_2d[index]['11'] = grid.distribution[x_range, y_range].sum()
                constraints_2d[index]['10'] = grid.distribution[x_range, y_complement_l].sum() + grid.distribution[x_range, y_complement_r].sum()
                constraints_2d[index]['01'] = grid.distribution[x_complement_l, y_range].sum() + grid.distribution[x_complement_r, y_range].sum()
                constraints_2d[index]['00'] = grid.distribution[x_complement_l, y_complement_l].sum() + grid.distribution[x_complement_l, y_complement_r].sum() + grid.distribution[x_complement_r, y_complement_l].sum() + grid.distribution[x_complement_r, y_complement_r].sum()
            # initialization
            distribution = np.zeros((2 ** len(attribute_indexes), len(attribute_indexes) + 1))
            i = 0
            for x in range(len(attribute_indexes) + 1):
                a = np.zeros(len(attribute_indexes))
                a[:x] = 1
                perms = set(itertools.permutations(a))
                for b in perms:
                    b = np.array(list(map(int, b)))
                    distribution[i][:-1] = b
                    distribution[i][-1] = 1 / np.size(distribution)
                    i += 1
            answer = distribution[-1, -1]
            # weighted updating
            old_answer = answer
            while True:
                # 1st. update according to 2-d constraints
                for key, grid_freq in constraints_2d.items():
                    key_index = [np.argwhere(attribute_indexes == key[0])[0,0], np.argwhere(attribute_indexes == key[1])[0,0]]
                    index11 = np.where((distribution[:,key_index] == [1,1]).all(axis = 1))[0]
                    index10 = np.where((distribution[:,key_index] == [1,0]).all(axis = 1))[0]
                    index01 = np.where((distribution[:,key_index] == [0,1]).all(axis = 1))[0]
                    index00 = np.where((distribution[:,key_index] == [0,0]).all(axis = 1))[0]
                    if np.sum(distribution[index11, -1]) != 0:
                        distribution[index11, -1] = distribution[index11, -1] / np.sum(distribution[index11, -1]) * grid_freq['11']
                    if np.sum(distribution[index10, -1]) != 0:
                        distribution[index10, -1] = distribution[index10, -1] / np.sum(distribution[index10, -1]) * grid_freq['10']
                    if np.sum(distribution[index01, -1]) != 0:
                        distribution[index01, -1] = distribution[index01, -1] / np.sum(distribution[index01, -1]) * grid_freq['01']
                    if np.sum(distribution[index00, -1]) != 0:
                        distribution[index00, -1] = distribution[index00, -1] / np.sum(distribution[index00, -1]) * grid_freq['00']                 
                # 2nd. update according to 1-d constraints
                for i, grid_freq in enumerate(constraints_1d):
                    index1 = np.where(distribution[:,i] == 1)[0]
                    index0 = np.where(distribution[:,i] == 0)[0]
                    if self.scheme == 'multiply':
                        if np.sum(distribution[index1, -1]) != 0:
                            distribution[index1, -1] = distribution[index1, -1] / np.sum(distribution[index1, -1]) * grid_freq[1]
                        if np.sum(distribution[index0, -1]) != 0:
                            distribution[index0, -1] = distribution[index0, -1] / np.sum(distribution[index0, -1]) * grid_freq[0]
                    elif self.scheme == 'add':
                        distribution[index1, -1] = self.__weighted_averaging(distribution[index1,-1], grid_freq[1])
                        distribution[index0, -1] = self.__weighted_averaging(distribution[index0,-1], grid_freq[0])
                distribution[:,-1] = distribution[:,-1] / np.sum(distribution[:, -1])
                answer = distribution[-1,-1]
                if abs(old_answer - answer) < 1 / (self.user_num ** 2):
                    break
                old_answer = answer
        return answer