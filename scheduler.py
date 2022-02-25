'''
scheduler is a python class to change values based on progression metrics.
Copyright (C) 2020 John T LaMaster

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

import math

from types import SimpleNamespace

__all__ = ['citation', 'scheduler']

citation = {'Source code': {'Author': 'Ing. John T LaMaster',
							'Date': 'October 2020'}}

class scheduler():
    def __init__(self, param, starting_epoch, total_epochs, total_batches, method='cosine', **kwargs):
        self.nBatch = total_batches - 1
        self.n_epochs = total_epochs
        self.init_epoch = starting_epoch
        if not isinstance(param, list): param = [param]
        self.initial_value = param
        self.method = method
        self.total = self.n_epochs * self.nBatch
        self.current_values = copy.copy(param)

        self.opt = SimpleNamespace(**kwargs)

    def __call__(self, epoch, batch):
        if epoch>=self.n_epochs + self.init_epoch:
            return self.current_values
        else:
            if self.method=='cosine':
                return self.cosine(epoch, batch)
            elif self.method=='linear':
                return self.linear(epoch, batch)
            elif self.method=='multiplicative':
                return self.multiplicative(epoch, batch)
            elif self.method=='step':
                return self.step(epoch, batch)
            elif self.method=='multistep':
                return self.multistep(epoch, batch)
            elif self.method=='exponential':
                return self.exponential(epoch, batch)


    def cosine(self, epoch, batch):
        if epoch < self.init_epoch:
            return [0 for _ in self.initial_value]
        elif epoch <= self.n_epochs + self.init_epoch:
            epoch = epoch - self.init_epoch
            current = (epoch % self.n_epochs) * self.nBatch + batch
            delta = current / self.total
            self.current_values = [0.5 * v * (1 + math.cos(math.pi * delta)) for v in self.initial_value]
            return self.current_values

    def linear(self, epoch, batch):
        if epoch < self.init_epoch:
            return [0] * len(self.initial_value)
        elif epoch <= self.n_epochs + self.init_epoch:
            epoch = epoch - self.init_epoch
            current = (epoch % self.n_epochs) * self.nBatch + batch
            delta = current / self.total
            self.current_values = [v * delta for v in self.initial_value]
            return self.current_values
        elif epoch >= self.n_epochs + self.init_epoch:
            return self.initial_value

    def multiplicative(self, epoch, batch):
        if epoch >= self.n_epochs + self.init_epoch:
            return self.initial_value
        elif batch==0 and epoch >= 1:
            self.current_values = [v * self.opt.gamma for v in self.initial_value]
            return self.current_values

    def step(self, epoch, batch):
        if batch==0 and epoch % self.opt.step_size == 0 and epoch >= 1:
            self.current_values = [v * self.opt.gamma for v in self.initial_value]
            return self.current_values
        else:
            return self.current_values

    def multistep(self, epoch, batch):
        if batch==0 and epoch in self.opt.milestones:
            self.current_values = [v * self.opt.gamma for v in self.initial_value]
            return self.current_values
        else:
            return self.current_values

    def exponential(self, epoch, batch):
        if epoch==0:
            return self.initial_value
        elif batch==0 and epoch >= 1:
            self.current_values *= self.opt.gamma
            return self.current_values
        else:
            return self.current_values
