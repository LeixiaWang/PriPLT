import numpy as np
import math
from parameters import arguments
import scipy
from numpy import True_, linalg as LA


class Frequency_oracle(object):
    '''
    the fundamental ldp frequency oracle
    '''
    def __init__(self, data, frequency_oracle_name = None, epsilon = arguments.epsilon, domain_size=None, merged_domain = None):
        '''
        Args:
            data(list): given users' data
            frequency_oracle_name(str): selected frequency oracle, when it's None, we choose it according epsilon
            epsilon(float): the privacy budget
            domain_size(int): the original domain_size
            merged_domain(nested list): the inner list indicates the indexes of the merged data
        '''
        self.user_number = len(data)
        self.domain_size = domain_size
        self.epsilon = epsilon
        self.real_count = np.zeros(self.domain_size, dtype = int)
        self.perturbed_count = np.zeros(self.domain_size, dtype = int)
        self.aggregated_count = np.zeros(self.domain_size, dtype = int) # after correction
        self.original_real_count = self._count(data)
        self.data = data
        self.set_merged_domain(merged_domain)
        self.set_frequency_oracle(frequency_oracle_name)
        self.set_privacy_budget()

    def set_privacy_budget(self):
        # set perturbation probability
        merged_domain_size = self.merged_domain_size[0] * self.merged_domain_size[1] if isinstance(self.merged_domain_size, tuple) else self.merged_domain_size
        if self.frequency_oracle_name in ['GRR','RR']:
            self.p = math.exp(self.epsilon) / (math.exp(self.epsilon)  + merged_domain_size - 1)
            self.q = 1.0 / (math.exp(self.epsilon) + merged_domain_size - 1)
        elif self.frequency_oracle_name in ['OUE', 'ORR']:
            self.p = 0.5
            self.q = 1.0 / (math.exp(self.epsilon) + 1)
        elif self.frequency_oracle_name in ['SW']:
            self.b = math.floor((self.epsilon * math.exp(self.epsilon) -  math.exp(self.epsilon) + 1)/(2 * math.exp(self.epsilon) * (math.exp(self.epsilon) - 1 - self.epsilon)) * merged_domain_size)
            self.p = math.exp(self.epsilon) / ((2 * self.b + 1) * math.exp(self.epsilon) + merged_domain_size - 1)
            self.q = 1 / ((2 * self.b + 1) * math.exp(self.epsilon) + merged_domain_size - 1)
            

    def set_merged_domain(self, merged_domain:list):
        if isinstance(self.domain_size, int):
            self.merged_domain_size = len(merged_domain) if merged_domain is not None else self.domain_size
            self.merged_domain_size = 2 if self.merged_domain_size == 1 else self.merged_domain_size
        else:
            self.merged_domain_size = (len(merged_domain[0]), len(merged_domain[1])) if merged_domain is not None else self.domain_size
        self.merged_real_count = self._merge_count(self.original_real_count, merged_domain).astype(int)

    def set_frequency_oracle(self, frequency_oracle_name:str):
        assert frequency_oracle_name in ['GRR', 'OUE', 'RR', 'ORR', 'SW', None]
        merged_domain_size = self.merged_domain_size[0] * self.merged_domain_size[1] if isinstance(self.merged_domain_size, tuple) else self.merged_domain_size
        if frequency_oracle_name is None:
            if merged_domain_size < 3 * math.exp(self.epsilon) + 2:
                self.frequency_oracle_name = 'GRR'
            else:
                self.frequency_oracle_name = 'OUE'
        else:
            self.frequency_oracle_name = frequency_oracle_name

    def _count(self, data:list):
        '''
        comput the count of each value in give data
        Returns:
            hist: the counts, i.e., the histogram
        '''
        if isinstance(self.domain_size, int):
            hist = np.zeros(self.domain_size)
            unique, counts = np.unique(data, return_counts=True)
            for i in range(len(unique)):
                hist[int(unique[i])] = counts[i]
        else:
            hist = np.zeros(self.domain_size) # only for 2-d
            unique, counts = np.unique(data, axis = 0, return_counts = True)
            for i in range(len(unique)):
                hist[*unique[i]] = counts[i]
        return hist
    
    def _merge_count(self, original_count:np.array, merged_domain:list):
        '''
        merge the histogram to the one with the merged domain
        '''
        if merged_domain is None:
            merged_count = original_count.copy()
        elif isinstance(self.domain_size, int):
            # 1-d array
            merged_count = np.zeros(len(merged_domain))
            for i in range(len(merged_domain)):
                merged_count[i] = np.sum(original_count[merged_domain[i]])
        else:
            # grid
            merged_count = np.zeros((len(merged_domain[0]),len(merged_domain[1])))
            for i in range(len(merged_domain[0])):
                x_range_l = merged_domain[0][i][0]
                x_range_r = merged_domain[0][i][1]+1
                for j in range(len(merged_domain[1])):
                    y_range_l = merged_domain[1][j][0]
                    y_range_r = merged_domain[1][j][1]+1
                    merged_count[i,j] = original_count[x_range_l : x_range_r, y_range_l : y_range_r].sum()                 
        return merged_count

    def _aggregate_values(self):
        '''
        estimate values in merged domain
        '''
        # perturbed count
        perturbed_count = np.zeros(self.merged_domain_size, dtype=int)
        perturbed_count += np.random.binomial(self.merged_real_count, self.p)
        perturbed_count += np.random.binomial(self.user_number - self.merged_real_count, self.q)
        # calibrated count
        a = 1.0 / (self.p - self.q)
        b = self.user_number * self.q / (self.p - self.q)
        calibrated_count = a * perturbed_count - b
        return calibrated_count

    def _aggregate_a_value(self):
        '''
        estimate a value that merged domain specified
        '''
        # perturbed count
        perturbed_count = np.random.binomial(self.merged_real_count[0], self.p)
        perturbed_count += np.random.binomial(self.user_number - self.merged_real_count[0], self.q)
        # calibrate count
        a = 1.0 / (self.p - self.q)
        b = self.user_number * self.q / (self.p - self.q)
        calibrated_count = a * perturbed_count - b
        return calibrated_count
    
    def _aggregate_distribution_sw(self, smooth = True):
        '''
        estimate the distribution via square wave
        '''
        return self._sw(self.data, 0, self.domain_size-1, self.epsilon, self.domain_size, self.domain_size, smooth)
        
    def get_real_count(self):
        # return self.merged_real_count
        return self.original_real_count
    
    def get_real_frequency(self):
        return self.get_real_count() / self.user_number

    def get_aggregated_count(self, smooth = True):        
        # perturb & estimate
        if self.frequency_oracle_name in ['GRR', 'OUE']:
            calibrated_count = self._aggregate_values()
        elif self.frequency_oracle_name in ['RR', 'ORR']:
            calibrated_count = self._aggregate_a_value()
        elif self.frequency_oracle_name == 'SW':
            if smooth is None:
                hist_smooth, hist_unsmooth = self._aggregate_distribution_sw(smooth)
                calibrated_count = (hist_smooth * self.user_number, hist_unsmooth * self.user_number)
            else:
                calibrated_count = self._aggregate_distribution_sw(smooth)
        return calibrated_count
        
    def get_aggregated_frequency(self, smooth = True):
        if smooth is None:
            hist_smooth, hist_unsmooth = self._aggregate_distribution_sw(smooth)
            return hist_smooth, hist_unsmooth
        else:
            calibrated_frequency = self.get_aggregated_count(smooth) / self.user_number
            return calibrated_frequency

    def get_one_theoretical_square_error(self, frequency = True):
        if frequency:
            var = self.q * (1 - self.q) / (self.user_number * (self.p - self.q) ** 2)
        else:
            var = self.user_number * self.q * (1 - self.q) / (self.p - self.q) ** 2
        if self.frequency_oracle_name == 'SW':
            var /= (2 * self.b) **2
        return var
  
    
    def _sw(self, ori_samples, l, h, eps, randomized_bins=1024, domain_bins=1024, smooth = True):
        ee = np.exp(eps)
        w = ((eps * ee) - ee + 1) / (2 * ee * (ee - 1 - eps)) * 2
        p = ee / (w * ee + 1)
        q = 1 / (w * ee + 1)

        samples = (ori_samples - l) / (h - l)
        randoms = np.random.uniform(0, 1, len(samples))

        noisy_samples = np.zeros_like(samples)

        # report
        index = randoms <= (q * samples)
        noisy_samples[index] = randoms[index] / q - w / 2
        index = randoms > (q * samples)
        noisy_samples[index] = (randoms[index] - q * samples[index]) / p + samples[index] - w / 2
        index = randoms > q * samples + p * w
        noisy_samples[index] = (randoms[index] - q * samples[index] - p * w) / q + samples[index] + w / 2

        # report matrix
        m = randomized_bins
        n = domain_bins
        m_cell = (1 + w) / m
        n_cell = 1 / n

        transform = np.ones((m, n)) * q * m_cell
        for i in range(n):
            left_most_v = (i * n_cell)
            right_most_v = ((i + 1) * n_cell)

            ll_bound = int(left_most_v / m_cell)
            lr_bound = int((left_most_v + w) / m_cell)
            rl_bound = int(right_most_v / m_cell)
            rr_bound = int((right_most_v + w) / m_cell)

            ll_v = left_most_v - w / 2
            rl_v = right_most_v - w / 2
            l_p = ((ll_bound + 1) * m_cell - w / 2 - ll_v) * (p - q) + q * m_cell
            r_p = ((rl_bound + 1) * m_cell - w / 2 - rl_v) * (p - q) + q * m_cell
            if rl_bound > ll_bound:
                transform[ll_bound, i] = (l_p - q * m_cell) * ((ll_bound + 1) * m_cell - w / 2 - ll_v) / n_cell * 0.5 + q * m_cell
                transform[ll_bound + 1, i] = p * m_cell - (p * m_cell - r_p) * (rl_v - ((ll_bound + 1) * m_cell - w / 2)) / n_cell * 0.5
            else:
                transform[ll_bound, i] = (l_p + r_p) / 2
                transform[ll_bound + 1, i] = p * m_cell

            lr_v = left_most_v + w / 2
            rr_v = right_most_v + w / 2
            r_p = (rr_v - (rr_bound * m_cell - w / 2)) * (p - q) + q * m_cell
            l_p = (lr_v - (lr_bound * m_cell - w / 2)) * (p - q) + q * m_cell
            if rr_bound > lr_bound:
                if rr_bound < m:
                    transform[rr_bound, i] = (r_p - q * m_cell) * (rr_v - (rr_bound * m_cell - w / 2)) / n_cell * 0.5 + q * m_cell

                transform[rr_bound - 1, i] = p * m_cell - (p * m_cell - l_p) * ((rr_bound * m_cell - w / 2) - lr_v) / n_cell * 0.5
                
            else:
                transform[rr_bound, i] = (l_p + r_p) / 2
                transform[rr_bound - 1, i] = p * m_cell

            if rr_bound - 1 > ll_bound + 2:
                transform[ll_bound + 2: rr_bound - 1, i] = p * m_cell

        max_iteration = 10000
        loglikelihood_threshold = 1e-3
        ns_hist, _ = np.histogram(noisy_samples, bins=randomized_bins, range=(-w / 2, 1 + w / 2))
        if smooth is None:
            hist_smooth = self._EMS(n, ns_hist, transform, max_iteration, loglikelihood_threshold)
            hist_unsmooth = self._EM(n, ns_hist, transform, max_iteration, loglikelihood_threshold)
            return hist_smooth, hist_unsmooth
        elif smooth:
            return self._EMS(n, ns_hist, transform, max_iteration, loglikelihood_threshold) * len(ori_samples)
        else:
            return self._EM(n, ns_hist, transform, max_iteration, loglikelihood_threshold) * len(ori_samples)
        

    def _EMS(self, n, ns_hist, transform, max_iteration, loglikelihood_threshold):
        # smoothing matrix
        smoothing_factor = 2
        binomial_tmp = [scipy.special.binom(smoothing_factor, k) for k in range(smoothing_factor + 1)]
        smoothing_matrix = np.zeros((n, n))
        central_idx = int(len(binomial_tmp) / 2)
        for i in range(int(smoothing_factor / 2)):
            smoothing_matrix[i, : central_idx + i + 1] = binomial_tmp[central_idx - i:]
        for i in range(int(smoothing_factor / 2), n - int(smoothing_factor / 2)):
            smoothing_matrix[i, i - central_idx: i + central_idx + 1] = binomial_tmp
        for i in range(n - int(smoothing_factor / 2), n):
            remain = n - i - 1
            smoothing_matrix[i, i - central_idx + 1:] = binomial_tmp[: central_idx + remain]
        row_sum = np.sum(smoothing_matrix, axis=1)
        smoothing_matrix = (smoothing_matrix.T / row_sum).T

        # EMS
        theta = np.ones(n) / float(n)
        theta_old = np.zeros(n)
        r = 0
        sample_size = sum(ns_hist)
        old_logliklihood = 0

        while LA.norm(theta_old - theta, ord=1) > 1 / sample_size and r < max_iteration:
            theta_old = np.copy(theta)
            X_condition = np.matmul(transform, theta_old)

            TMP = transform.T / X_condition

            P = np.copy(np.matmul(TMP, ns_hist))
            P = P * theta_old

            theta = np.copy(P / sum(P))

            # Smoothing step
            theta = np.matmul(smoothing_matrix, theta)
            theta = theta / sum(theta)

            logliklihood = np.inner(ns_hist, np.log(np.matmul(transform, theta)))
            imporve = logliklihood - old_logliklihood

            if r > 1 and abs(imporve) < loglikelihood_threshold:
                # print("stop when", imporve / old_logliklihood, loglikelihood_threshold)
                break

            old_logliklihood = logliklihood

            r += 1
        return theta


    def _EM(self, n, ns_hist, transform, max_iteration, loglikelihood_threshold):
        theta = np.ones(n) / float(n)
        theta_old = np.zeros(n)
        r = 0
        sample_size = sum(ns_hist)
        old_logliklihood = 0

        while LA.norm(theta_old - theta, ord=1) > 1 / sample_size and r < max_iteration:
            theta_old = np.copy(theta)
            X_condition = np.matmul(transform, theta_old)

            TMP = transform.T / X_condition

            P = np.copy(np.matmul(TMP, ns_hist))
            P = P * theta_old

            theta = np.copy(P / sum(P))

            logliklihood = np.inner(ns_hist, np.log(np.matmul(transform, theta)))
            imporve = logliklihood - old_logliklihood

            if r > 1 and abs(imporve) < loglikelihood_threshold:
                # print("stop when", imporve, loglikelihood_threshold)
                break

            old_logliklihood = logliklihood

            r += 1
        return theta

if __name__ == '__main__':
    pass