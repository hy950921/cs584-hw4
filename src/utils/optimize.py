from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from six.moves import range


def baseline_als(self):

    bu = np.zeros(self.trainset.n_users)
    bi = np.zeros(self.trainset.n_items)

    # cdef int u, i
    # cdef double r, err, dev_i, dev_u
    global_mean = self.trainset.global_mean

    n_epochs = self.bsl_options.get('n_epochs', 10)
    reg_u = self.bsl_options.get('reg_u', 15)
    reg_i = self.bsl_options.get('reg_i', 10)

    for dummy in range(n_epochs):
        for i in self.trainset.all_items():
            dev_i = 0
            for (u, r) in self.trainset.ir[i]:
                dev_i += r - global_mean - bu[u]

            bi[i] = dev_i / (reg_i + len(self.trainset.ir[i]))

        for u in self.trainset.all_users():
            dev_u = 0
            for (i, r) in self.trainset.ur[u]:
                dev_u += r - global_mean - bi[i]
            bu[u] = dev_u / (reg_u + len(self.trainset.ur[u]))

    return bu, bi


def baseline_sgd(self):

    bu = np.zeros(self.trainset.n_users)
    bi = np.zeros(self.trainset.n_items)

    # cdef int u, i
    # cdef double r, err
    global_mean = self.trainset.global_mean

    n_epochs = self.bsl_options.get('n_epochs', 20)
    reg = self.bsl_options.get('reg', 0.02)
    lr = self.bsl_options.get('learning_rate', 0.005)

    for dummy in range(n_epochs):
        for u, i, r in self.trainset.all_ratings():
            err = (r - (global_mean + bu[u] + bi[i]))
            bu[u] += lr * (err - reg * bu[u])
            bi[i] += lr * (err - reg * bi[i])

    return bu, bi