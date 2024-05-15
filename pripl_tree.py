import numpy as np
from parameters import *
from frequency_oracle import *
import pl_approximation
from treelib import Tree, Node
import queue
from collections import deque

class Node_pl(object):
    def __init__(self, interval_left, interval_right, domain_size, frequency = None, slope = None, error = None):
        self.interval_left = interval_left
        self.interval_right = interval_right
        self.set_data(frequency, slope, error, domain_size, comp_partial_query_weight = True)
        self.left_user_ratio = 0
        self.error_coef = None


    def set_data(self, frequency = None, slope = None, error = None, domain_size = None, comp_partial_query_weight = False):
        if frequency is not None:
            self.frequency = frequency
        if slope is not None:
            self.original_slope = slope
            self.slope = slope
        if error is not None:
            self.error = error
        if comp_partial_query_weight:
            self.partial_query_weight = 2 * (self.interval_left + 1) * (domain_size - self.interval_right) / (domain_size * (domain_size - 1))
    
    def reset_slope(self, slope):
        self.slope = slope 

    def initial_error_coef(self, nodes_id, nodes_number):
        self.error_coef = np.zeros(nodes_number)
        self.error_coef[nodes_id] = 1

    def set_allocated_users(self, total_num, left_num, proportion):
        '''
        as we allocate users according indexes from small to large, given left num and the porpotion we will take,\
        we can refer the users' ids for the node
        '''
        self.user_num = round(left_num * proportion)
        self.left_user_num = left_num - self.user_num
        index_l = total_num - left_num
        index_r =index_l + self.user_num - 1
        self.allocated_users = [index_l, index_r]


class Pripl_tree(object):
    '''
    the private piecewise linear tree, only for 1-d data
    '''
    def __init__(self, data, domain_size, epsilon = arguments.epsilon, alpha = arguments.alpha, search_granularity = arguments.search_granularity):
        '''
        Args:
            data(np.array): a 2-d ndarray with only 1-d data
            domain_size(int): the domain size of data
            epsilon(float): privacy budget
            error_metric(str): "max_absolute" or "square"
        '''
        assert np.ndim(data) == 1 or data.shape[1] == 1
        self.data = data.flatten()
        self.user_num = len(self.data)
        self.domain_size = int(domain_size)
        self.epsilon = epsilon
        self.alpha = alpha
        self.search_granularity = search_granularity
        self.update_error = True
        

    def build_pripl_tree(self):
        '''
        Args:
            height(int): when adpative is False, this parameter specify the height of the tree; or it decide the least height of the tree
            adaptive(boolean): whether we split tree according to the estimated node error
        '''
        # phase 1: pl-function fitting
        self._underlying_user_allocation()
        self.noisy_hist, self.noisy_hist_ref = self._estimate_underly_hist(self.underlying_users)
        segments, slopes = self._pl_fitting()
        # phase 2: pripl-tree construction
        pripl_tree = self._pripl_construction(segments, slopes)
        self._uniform_user_allocation(pripl_tree)
        for node in pripl_tree.all_nodes():
            if not node.is_root():
                self._estimate_node_frequency(node)
        # phase 3: pripl-tree refinement: tree consistency and slope refine
        # weighted average
        self._weighted_average(pripl_tree, consider_underlying_distribution=True)
        # frequency consistency
        self._frequency_consistency(pripl_tree)
        # slope refine
        for node in pripl_tree.leaves():
            d = node.data.interval_right - node.data.interval_left + 1
            if d > 1:
                node.data.reset_slope(self._refine_slope(node.data.original_slope, d, node.data.frequency))
        self.tree = pripl_tree
        return pripl_tree


    def refine_pripl_tree_leaves(self, frequencies, pripl_tree = None):
        if pripl_tree is None:
            pripl_tree = self.tree
        # update leaves
        leaves = self.get_leaves_in_order(pripl_tree)
        for k in range(len(leaves)):
            leaves[k].data.frequency = frequencies[k]
        # slope refine
        for node in leaves:
            d = node.data.interval_right - node.data.interval_left + 1
            if d > 1:
                node.data.reset_slope(self._refine_slope(node.data.original_slope, d, node.data.frequency))
        # update non-leaf nodes
        for node in self._dfs_postorder_traveral(pripl_tree):
            if not node.is_leaf():
                node.data.frequency = 0
                for child in pripl_tree.children(node.identifier):
                    node.data.frequency += child.data.frequency
    

    def get_distribution_from_tree(self, pripl_tree = None):
        if pripl_tree is None:
            pripl_tree = self.tree
        distr = np.zeros(self.domain_size)
        for node in pripl_tree.leaves():
            d = node.data.interval_right - node.data.interval_left + 1
            x = np.arange(node.data.interval_left, node.data.interval_right+1).astype(int)
            distr[x] = self._pl_range_query(x,x, node.data.interval_left, d, node.data.slope, node.data.frequency)
        # this negative frequency comes from the inaccurate float refined slopes
        distr[distr < 0] = 0
        return distr
        

    def query(self, query_range, pripl_tree = None):
        if pripl_tree is None:
            pripl_tree = self.tree
        ans = 0
        query_queue = queue.Queue()
        root = pripl_tree.get_node(pripl_tree.root)
        query_queue.put(root)
        while not query_queue.empty():
            node = query_queue.get()
            intersect_s = self.__intersection([node.data.interval_left, node.data.interval_right], query_range)
            if intersect_s > 0:
                ans += node.data.frequency
            elif intersect_s < 0:
                if node.is_leaf():
                    l = max(query_range[0], node.data.interval_left)
                    r = min(query_range[1], node.data.interval_right)
                    d = node.data.interval_right - node.data.interval_left + 1
                    ans += self._pl_range_query(l, r, node.data.interval_left, d, node.data.slope, node.data.frequency)
                else:
                    for child in pripl_tree.children(node.identifier):
                        query_queue.put(child)
        return ans


    def __intersection(self, node_range, query_range):
        if node_range[0] > query_range[1] or node_range[1] < query_range[0]:
            # no overlap
            return 0
        elif node_range[0] >= query_range[0] and node_range[1] <= query_range[1]:
            # complete overlap
            return 1
        else:
            # partial overlap
            return -1
        

    def _pl_range_query(self, l, r, segment_start, d, slope, count):
        '''
        Args:
            [l,r]: the bound of range query in the segment.
            segment_start: i.e. ceil(s_{i}). he start point of the segment.
            d: the domain size of this segment
            slope: the refined slope of this segment
            count: the sum of counts or frequencies of items in this segment
        '''
        return (r - l + 1) * (slope * ((l + r)/2 - segment_start - (d-1)/2) + count / d)


    def get_leaves_in_order(self, pripl_tree = None):
        if pripl_tree is None:
            pripl_tree = self.tree
        leaves = pripl_tree.leaves()
        intervals = np.zeros(len(leaves))
        for i in range(len(leaves)):
            intervals[i] = leaves[i].data.interval_left
        return np.array(leaves)[np.argsort(intervals)]

    
    def _underlying_user_allocation(self):
        num = int(self.user_num * self.alpha) - 1
        self.underlying_users = [0,num]
    

    def _uniform_user_allocation(self, pl_tree):
        tree_user_num = self.user_num - (self.underlying_users[1] - self.underlying_users[0] + 1)
        for node in self._bfs(pl_tree, return_root=True):
            if node.is_root():
                node.data.set_allocated_users(self.user_num, tree_user_num, 0)
            else:
                to_be_allocated_user_num = pl_tree.parent(node.identifier).data.left_user_num
                subtree_depth = pl_tree.subtree(node.identifier).depth() + 1
                node.data.set_allocated_users(self.user_num, to_be_allocated_user_num, 1/subtree_depth)


    def _estimate_underly_hist(self, users):
        data = self.data[users[0]:users[1]]
        # using sw
        fo = Frequency_oracle(data, frequency_oracle_name = 'SW', epsilon=self.epsilon, domain_size=self.domain_size)
        hist_SW, hist_SW_woS = fo.get_aggregated_frequency(None)
        return hist_SW, hist_SW_woS


    def _estimate_node_frequency(self, node):
        data = self.data[node.data.allocated_users[0]: node.data.allocated_users[1]+1]
        subdomain = range(node.data.interval_left, node.data.interval_right+1)
        fo = Frequency_oracle(data, frequency_oracle_name='ORR', epsilon=self.epsilon, domain_size=self.domain_size, merged_domain=[subdomain])
        node.data.frequency = fo.get_aggregated_frequency()
        node.data.error = fo.get_one_theoretical_square_error()
    

    def _norm_sub(self, est_dist, est_sum = 1):
        estimates = np.copy(est_dist)
        while (estimates < 0).any():
            estimates[estimates < 0] = 0
            total = np.sum(estimates)
            mask = estimates > 0
            diff = (est_sum - total) / np.sum(mask)
            estimates[mask] += diff
        return estimates
    

    def _pl_fitting(self, minima_segment_length = 1, maximal_segment_number = 32):
        '''
        Fitting the noisy histogram with k picewise linear function

        Error: RSS
        '''
        # define some important variables
        segments = np.array([[0, self.domain_size - 1]], dtype=int)
        breakpoints = np.array([0, self.domain_size - 1], dtype=float)
        split_signs = np.ones(len(segments))
        segment_length = np.array([self.domain_size])
        var = 4 * math.exp(self.epsilon) / (self.user_num * (math.exp(self.epsilon) - 1) ** 2)
        part = 2
        # the loop to split the segment until reach the termination condition
        for noisy_hist in [self.noisy_hist_ref, self.noisy_hist]:
            slopes = pl_approximation.multi_piece_closed_solution(noisy_hist, breakpoints, False)
            segment_error = np.array(pl_approximation.RSS(noisy_hist, breakpoints, slopes, False))
            last_loss = np.sum(segment_error)
            split_condition = True
            while np.any(split_signs == 1) and split_condition:
                error_list = segment_error * split_signs
                index = np.argmax(error_list)
                best_break_point, breakpoints, slopes = pl_approximation.search_with_multi_level_steps(noisy_hist, breakpoints, segments[index][0], segments[index][1], True, minima_segment_length=minima_segment_length, support_searched_domain_size=self.search_granularity)
                # update the segments
                temp = segments.copy()
                segments = np.empty((len(breakpoints) - 1, 2), dtype=int)
                segments[:index] = temp[:index]
                segments[index] = [temp[index][0], math.floor(best_break_point)]
                segments[index+1] = [math.floor(best_break_point)+1, temp[index][1]]
                segments[index+2:] = temp[index+1:]
                # update the segment length
                segment_length = np.array([seg[1] - seg[0] + 1 for seg in segments])
                # update the spilt sign
                segment_fre = np.array([self.noisy_hist[seg[0]:seg[1]+1].sum() for seg in segments])
                split_signs = (segment_length >= 2 * minima_segment_length) & (segment_fre >= math.sqrt(var))
                # compute the segment error
                segment_error = np.array(pl_approximation.RSS(noisy_hist, breakpoints, slopes, False))
                # RSS
                total_loss = np.sum(segment_error)
                # Split condition
                split_condition = (round(last_loss / total_loss, 2) > 1) & (len(segments) < maximal_segment_number / part)
                last_loss = total_loss
            part -= 1
        slopes = slopes[1:]
        return segments, slopes
        

    def _pripl_construction(self, segments, slopes, branch = 2):
        '''
        Phase 2: PriPL Tree Construction

        step 1: we initially build a binary tree, where the optimized fan-out for non-leaf nodes is 2
        step 2: we adaptively reduce some non-leaf node in the tree to optimized the total range query error. 
            After this step, we derive a unbalanced tree, and each node may have different fan-out.
        '''
        # step 1: the b-ary tree
        pripl_tree = Tree()
        node_queue = queue.Queue()
        root = Node(tag='root', identifier=0, data=Node_pl(interval_left=0, interval_right=self.domain_size-1, domain_size=self.domain_size, frequency=1, error=0))
        pripl_tree.add_node(root)
        if len(segments) == 1:
            root.data.set_data(slope = slopes[0])
        else:
            # initial all leaves
            for i, seg in enumerate(segments):
                node_data = Node_pl(seg[0], seg[1], self.domain_size, frequency=self.noisy_hist[seg[0]:seg[1]+1].sum(), slope = slopes[i])
                pripl_tree.create_node(tag='n_'+str(int(i+1)), identifier = i+1, parent=root, data=node_data)
                node = pripl_tree.get_node(i+1)
                node_queue.put(node)
            # generate non-leaf nodes
            node_id = len(segments)
            to_be_proc_num = len(segments)
            parent_num = math.ceil(to_be_proc_num / branch) if branch > 2 or to_be_proc_num % branch != 1 else math.floor(to_be_proc_num / branch)
            avg_seg_length = self.domain_size / parent_num
            while parent_num > 1:
                while parent_num != 0:
                    children_num = math.floor(to_be_proc_num / parent_num)
                    children = [node_queue.get() for _ in range(children_num)]
                    if to_be_proc_num % parent_num != 0:
                        adaptive = node_queue.queue[0]
                        reject_adap = children[-1].data.interval_right - children[0].data.interval_left + 1
                        accept_adap = reject_adap + adaptive.data.interval_right - adaptive.data.interval_left + 1
                        if (accept_adap - avg_seg_length) < (reject_adap - avg_seg_length) or parent_num == 1:
                            children.append(node_queue.get())
                    node_id += 1
                    pripl_tree.create_node(tag='n_'+str(int(node_id)), identifier=node_id, parent=root)
                    accum_fre = 0
                    for child in children:
                        accum_fre += child.data.frequency
                        pripl_tree.move_node(child.identifier, node_id)
                    parent_data = Node_pl(children[0].data.interval_left, children[-1].data.interval_right, self.domain_size, frequency=accum_fre)
                    parent = pripl_tree.get_node(node_id)
                    parent.data = parent_data
                    node_queue.put(parent)
                    parent_num -= 1
                    to_be_proc_num -= children_num
                to_be_proc_num = node_queue.qsize()
                parent_num = math.ceil(to_be_proc_num / branch) if branch > 2 or to_be_proc_num % branch != 1 else math.floor(to_be_proc_num / branch)
                avg_seg_length = self.domain_size / parent_num
        # # step 2: reduce some non-leaf nodes by DFS
        for node in self._dfs_postorder_traveral(pripl_tree):
            if not node.is_leaf() and not node.is_root():
                if self._reduce_determine(pripl_tree, node):
                    pripl_tree.link_past_node(node.identifier)
        return pripl_tree


    def _reduce_determine(self, pripl_tree, node):
        pripl_tree_reduced = Tree(pripl_tree.subtree(pripl_tree.root), deep=True)
        pripl_tree_reduced.link_past_node(node.identifier)
        error_remain = error_reduce = 0
        anc_left_ratio_remain = anc_left_ratio_reduce = 1
        for anc_id in list(pripl_tree.rsearch(node.identifier))[-2::-1]:
            if anc_id != node.identifier:
                l_reduce = pripl_tree_reduced.subtree(anc_id).depth() + 1
                anc_reduce = pripl_tree_reduced.get_node(anc_id)
                error_reduce += l_reduce / anc_left_ratio_reduce * (anc_reduce.data.partial_query_weight - pripl_tree_reduced.parent(anc_id).data.partial_query_weight)
                anc_left_ratio_reduce = anc_left_ratio_reduce * (1 - 1 / l_reduce)
            l_remain = pripl_tree.subtree(anc_id).depth() + 1
            anc_remain = pripl_tree.get_node(anc_id)
            error_remain += l_remain / anc_left_ratio_remain * (anc_remain.data.partial_query_weight - pripl_tree.parent(anc_id).data.partial_query_weight)
            anc_left_ratio_remain = anc_left_ratio_remain * (1 - 1 / l_remain)
        pripl_tree_reduced.get_node(pripl_tree.parent(node.identifier).identifier).data.left_user_ratio = anc_left_ratio_reduce
        node.data.left_user_ratio = anc_left_ratio_remain
        for des in self._bfs(pripl_tree.subtree(node.identifier), return_root=False):
            if des.identifier == node.identifier:
                continue
            l_reduce = pripl_tree_reduced.subtree(des.identifier).depth() + 1
            anc_left_ratio_reduce = pripl_tree_reduced.parent(des.identifier).data.left_user_ratio
            error_reduce += l_reduce / anc_left_ratio_reduce * (des.data.partial_query_weight - pripl_tree_reduced.parent(des.identifier).data.partial_query_weight)
            pripl_tree_reduced.get_node(des.identifier).data.left_user_ratio = anc_left_ratio_reduce * (1 - 1 / l_reduce)
            l_remain = pripl_tree.subtree(des.identifier).depth() + 1
            anc_left_ratio_remain = pripl_tree.parent(des.identifier).data.left_user_ratio
            error_remain += l_remain / anc_left_ratio_remain * (des.data.partial_query_weight - pripl_tree.parent(des.identifier).data.partial_query_weight)
            des.data.left_user_ratio = anc_left_ratio_remain * (1 - 1 / l_remain)
        return error_reduce < error_remain


    def _weighted_average(self, pripl_tree, consider_underlying_distribution = True):
        '''
        dfs: postorder traversal, implemented through tow stacks
        '''
        # global parameter
        nodes_num = len(pripl_tree.all_nodes()) - 1
        self.error_vector = np.zeros(nodes_num)
        # root node
        root = pripl_tree.get_node(pripl_tree.root)
        root.data.error_coef = np.zeros(nodes_num)
        # trasversal of all nodes
        node_index = 0
        for node in self._dfs_postorder_traveral(pripl_tree):
            node.data.initial_error_coef(node_index, nodes_num)
            if node.is_leaf():
                if consider_underlying_distribution:
                    self._update_frequency_in_wa(pripl_tree, node, is_leaf = True)
                self.error_vector[node_index] = node.data.error
            else:
                self.error_vector[node_index] = node.data.error
                alpha_1, alpha_2 = self._update_frequency_in_wa(pripl_tree, node, is_leaf = False)
                update_error_coef = alpha_1 * node.data.error_coef
                for child in pripl_tree.children(node.identifier):
                    update_error_coef += alpha_2 * child.data.error_coef
                node.data.error_coef = update_error_coef.copy()
                node.data.error = np.dot(update_error_coef, self.error_vector)
            node_index += 1


    def _update_frequency_in_wa(self, pripl_tree, node, is_leaf):
        x_1 = node.data.frequency
        var_1 = node.data.error
        x_2 = 0
        var_2 = 0
        if is_leaf:
            x_2 = self.noisy_hist[node.data.interval_left: node.data.interval_right+1].sum()
            var_2 = np.square(x_2 - x_1)
            var_2 = var_2 - var_1 if var_2 > var_1 else var_2
            var_2 = min(var_2, np.square(x_2))
        else:
            for child in pripl_tree.children(node.identifier):
                x_2 += child.data.frequency
                var_2 += child.data.error
        alpha = var_2 / (var_1 + var_2)
        x = alpha * x_1 + (1 - alpha) * x_2
        node.data.frequency = x
        node.data.error = var_1 * var_2 / (var_1 + var_2)
        return alpha, 1 - alpha


    def _frequency_consistency(self, pripl_tree, non_negative = True):
        '''
        bfs: implemented through queue
        '''
        for node in self._bfs(pripl_tree, return_root = True):
            if not node.is_leaf():
                parent_f = node.data.frequency
                children = pripl_tree.children(node.identifier)
                children_f = np.zeros(len(children))
                error_coef_matrix = np.zeros((len(children), len(pripl_tree.all_nodes())-1))
                for i in range(len(children)):
                    children_f[i] = children[i].data.frequency
                    error_coef_matrix[i] = children[i].data.error_coef
                update_children_f = self._update_frequency_in_fc(children_f, parent_f, non_negative)
                C_plus = np.where(update_children_f > 0)[0]
                coef = np.zeros(len(children))
                if len(C_plus) > 0:
                    coef[C_plus] = - 1 / len(C_plus)
                for i in range(len(children)):
                    children[i].data.frequency = update_children_f[i]
                    if self.update_error:
                        if i in C_plus: # f_c in C+, then we update its variance
                            coef_w = coef.copy()
                            coef_w[i] += 1
                            children[i].data.error_coef = (1 / len(C_plus)) * node.data.error_coef + np.dot(coef_w, error_coef_matrix)
                            children[i].data.error = np.dot(np.square(children[i].data.error_coef), self.error_vector)


    def _update_frequency_in_fc(self, children_f:np.array, parent_f, non_negative = True):
        # we can not compute this directly
        if parent_f == 0:
            C = np.zeros(np.shape(children_f))
        else:
            C = children_f.copy()
            C = C - (C.sum() - parent_f) / np.size(C)
            if non_negative:
                while (C < 0).any():
                    C[C < 0] = 0
                    mask = (C > 0) 
                    C[mask] += (parent_f - C.sum()) / np.sum(mask)
        return C

    def _weighted_update_frequency_in_fc(self, children_f:np.array, children_var:np.array, parent_f, non_negative = True, return_coef = False):
        # we can not compute this directly
        if parent_f == 0:
            C = np.zeros(np.shape(children_f))
            W = np.zeros(np.shape(children_f))
        else:
            C = children_f.copy()
            W = children_var / np.sum(children_var)
            C = C - (C.sum() - parent_f) * W
            if non_negative:
                while (C < 0).any():
                    mask_p = (C > 0) 
                    mask_0 = (C <= 0)
                    W[mask_0] = 0
                    W[mask_p] = children_var[mask_p] / np.sum(children_var[mask_p])
                    C = children_f.copy()
                    C[mask_0] = 0
                    C = C - (C.sum() - parent_f) * W
        if return_coef:
            return C, W
        else:
            return C

    def _refine_slope(self, slope, d, count):
        assert d > 1   
        if count == 0:
            refined_slope = 0
        else:  
            refined_slope = slope
            bound = 2 * count / (d * (d - 1))
            if refined_slope > bound:
                refined_slope = bound
            elif refined_slope < - bound:
                refined_slope = - bound
        return refined_slope
    

    def _dfs_postorder_traveral(self, pripl_tree):
        # we do not return root, which does not store any estimate value
        stack_travl = deque()
        stack_visit = deque()
        root = pripl_tree.get_node(pripl_tree.root)
        for child in pripl_tree.children(root.identifier):
            stack_travl.append(child)
        while len(stack_travl) != 0:
            node = stack_travl[-1]
            if node.is_leaf():
                yield stack_travl.pop()
            else:
                if len(stack_visit) > 0 and node is stack_visit[-1]:
                    yield stack_travl.pop()
                    stack_visit.pop()
                else:
                    stack_visit.append(node)
                    for child in pripl_tree.children(node.identifier):
                        stack_travl.append(child)


    def _bfs(self, pripl_tree, return_root = False):
        travl_queue = queue.Queue()
        root = pripl_tree.get_node(pripl_tree.root)
        travl_queue.put(root)
        while not travl_queue.empty():
            node = travl_queue.get()
            if not node.is_leaf():
                for child in pripl_tree.children(node.identifier):
                    travl_queue.put(child)
            if return_root or not node.is_root():
                yield node