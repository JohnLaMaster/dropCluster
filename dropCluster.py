from collections import Counter, OrderedDict
from random import sample

import numpy as np
import torch
import torch.nn as nn
from modules.aux.auxiliary import is_leaf
from modules.cluster.cluster_detection import GAP_STATISTICS
from numpy.linalg import norm as L2dist
from scipy.stats import beta
from sklearn.cluster import FeatureAgglomeration
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.preprocessing import LabelBinarizer

__all__ = ['citation', 'update_DropCluster', 'update_DropClusterProbability', 'DropCluster1d', 'DropCluster2d', 'DropCluster3d']


citation = OrderedDict({'dropCluster': {'Title': 'DropCluster: A structured dropout for convolutional networks',
                                        'Authors': 'Liyan Chen, Philip Gautier, Sergul Aydore',
                                        'Year': '2020',
                                        'Journal': 'ECCV',
                                        'Institution': 'Korea Advanced Institute of Science and Technology, Lunit Inc., and Adobe Research',
                                        'URL': 'https://arxiv.org/pdf/1807.06521.pdf',
                                        'Notes': 'Modified to be compatible with PyTorch and torch.autograd. Can be implemented as a module. '
                                                 'Als includes a function that can be called from the training script to both activate the dropCluster '
                                                 'block at a given epoch as well as to update the drop probability as mentioned in the publication. '
                                                 'I was not able to incorporate ReNA for clustering, so it is based on sklearn and feature agglomeration.',
                                        'Source Code': 'Modified from: https://github.com/miguelvr/dropblock/issues/30'},
                       'Spatial Hopkins Statistic':
                                        {'Title': 'Validating Clusters using the Hopkins Statistic',
                                        'Authors': 'Amit Banerjee, Rajesh N. Dave',
                                        'Year': '2004',
                                        'Journal': 'IEEE International Conference on Fuzzy Systems',
                                        'Institution': 'New Jersey Institute of Technology',
                                        'URL': 'https://ieeexplore.ieee.org/document/1375706/authors#authors',
                                        'Notes': 'Modified for use inside a PyTorch network',
                                        'Source Code': 'Modified from: https://datascience.stackexchange.com/questions/14142/cluster-tendency-using-hopkins-statistic-implementation-in-python, '
                                                       'https://matevzkunaver.wordpress.com/2017/06/20/hopkins-test-for-cluster-tendency/'}
                        'Source Code': 
                        				{'Author': 'Ing. John T LaMaster',
                        				'Date': 'September 2020'}})





def is_leaf(model):
    return get_num_gen(model.children()) == 0

def get_num_gen(gen):
    return sum(1 for x in gen)

def update_DropCluster(model, activate=False, prob=False):
    if activate:
        activate_DropCluster(model)
    if prob:
        update_DropClusterProbability(model, prob)

def activate_DropCluster(model):
    for m in model._modules:
        child = model._modules[m]
        if is_leaf(child):
            if isinstance(child, DropCluster1d) or \
               isinstance(child, DropCluster2d) or \
               isinstance(child, DropCluster3d):
                child._activated()
                child._prepare()
        else:
            update_DropCluster(child)
    return model


def update_DropClusterProbability(model, prob, dev=''):
    dev = model.device if dev=='' else dev
    for m in model._modules:
        child = model._modules[m]
        if is_leaf(child):
            if isinstance(child, DropCluster1d):
                setattr(child, 'p', torch.tensor(prob[0], requires_grad=False).to(dev))
        else:
            update_DropClusterProbability(child, prob, dev)
    return model


class DropCluster1d(nn.Module):
    # Modified from: https://github.com/miguelvr/dropblock/issues/30
    def __init__(self, p=0.2):
        super(DropCluster1d, self).__init__()
        self.register_buffer('p', torch.tensor(p))
        self.register_buffer('mask', torch.empty(0))
        self.register_buffer('channel_mask', torch.empty(0))

        self.cs = []
        self.activated = False
        self.prepare = False

    def extra_repr(self):
        return 'drop_prob=%.2f' % tuple([self.p.item()])

    def _activated(self):
        self.activated = True

    def _prepare(self):
        self.prepare = True

    def _mc_sampling(self):
        self.MC_testing = True

    def _cluster(self, input):
        bs, self.ch, self.feat = input.shape[0], input.shape[1], input.shape[2]
        self.mask = torch.zeros_like(input[0,::]).unsqueeze(0)
        input = input.numpy()

        # Boolean torch tensor indicating unstructured channels (threshold = 0.25, somewhat arbitrary)
        self.ind = self.SpatialHopkins(input)
        self.channel_mask = self._channel_mask

        # Number of clusters per channel
        self.k = [0.] * self.ch

        # Connectivity graph
        _knn_graph = grid_to_graph(n_x=self.feat, n_y=1, return_as=np.ndarray)

        for i in range(self.ch):
            # If not unstructured, then proceed
            if not self.ind[i]:
                # Determine the optimal number of clusters
                self.k[i] = GAP_STATISTICS(input[:,i,:],np.arange(2, self.feat.sqrt()))

                # Clustering algorithm
                clstr = FeatureAgglomeration(linkage='ward', connectivity=_knn_graph, n_clusters=self.k[i])
                clstr.fit(input[:,i,:])
                self.mask[i,::] = torch.from_numpy(clstr.labels_)
                self.cs[i].append(Counter(clstr.labels_))

        self.prepare = False

    def _binary_mask(self, s0): # s0 = batchSize
        # ones helps with counting and is reassigned below - more helpful than empty
        # numel and sum are used to normalize the outputs so that when no dropout is
        # applied, the values scale appropriately. Unstructured channels should not
        # be included because they will also not be used during inference
        # This version uses the same dropout mask for all samples in the batch
        bm = torch.ones((1, self.ch, self.feat))
        # bm[:,self.ind,:].fill_(0)
        bm *= self.channel_mask
        numel = bm.sum(-1) * s0
        for i in range(self.ch):
            div = []
            for _, v in self.cs[i].items():
                div.append(v)
            div = torch.FloatTensor(div)
            gamma = (1 - self.p) / div
            gamma *= self.feat / (self.feat - div + 1)
            mask = self._expand_mask(self.mask[i,::], self.k[i]).unsqueeze(0)
            prob = torch.rand(1,len(gamma))
            for n in range(len(gamma)):
                clusters = (prob[:,n] < gamma[n])
                mask[clusters,::].fill_(0)
            bm[:,i,:] = mask.sum(0).unsqueeze(0)
            bm = bm.repeat(s0)

        return bm, numel, torch.sum(bm,dim=2)

    def _binary_mask_v0(self, s0): # s0 = batchSize
        # This versions uses separate dropout masks for each sample in the batch
        # Good for running MC sampling of individual samples at test time
        bm = torch.ones((s0, self.ch, self.feat))
        bm *= self.channel_mask
        numel = bm.sum(-1)
        for i in range(self.ch):
            div = []
            for _, v in self.cs[i].items():
                div.append(v)
            div = torch.FloatTensor(div)
            gamma = (1 - self.p) / div
            gamma *= self.feat / (self.feat - div + 1)
            mask = self._expand_mask(self.mask[i,::], self.k[i]).unsqueeze(0).repeat(s0)
            prob = torch.rand(s0,len(gamma))
            for n in range(len(gamma)):
                clusters = (prob[:,n] < gamma[n])
                mask[clusters,::].fill_(0)
            bm[:,i,:] = mask.sum(0).unsqueeze(0)

        return bm, numel, torch.sum(bm, dim=2)

    def _expand_mask(self, mask, num_clusters):
        expanded = torch.empty_like(mask).unsqueeze(0).repeat(num_clusters)
        lb = LabelBinarizer()
        lb.fit(np.asarray(mask))
        for i in range(num_clusters):
            expanded[i,::] = torch.from_numpy(lb.transform([i]))
        return expanded

    @property
    def _channel_mask(self):
        mask = torch.ones_like(self.mask)
        return mask[self.ind,:].fill_(0).unsqueeze(0)

    def forward(self, x):
        if self.prepare:
            self._cluster(x)
            return x
        if not self.activated: return x
        if not self.training and not self.MC_testing: return x * self.channel_mask.repeat(x.shape[0])
        elif self.training and not self.MC_testing:
            bm, numel, summ = self._binary_mask(x.shape[0])
        elif self.MC_testing:
            bm, numel, summ = self._binary_mask_v0(x.shape[0])

        return x * bm * numel / summ

    def SpatialHopkins(self, input):
        H, n = SpatialHopkinsStatistic(input)
        # Reference: To Cluster, or Not to Cluster: An Analysis of Clusterability Methods (p.9)
        alpha = 0.05
        threshold = torch.from_numpy(beta.ppf(1-alpha, n, n))
        H = H.mean(dim=0)
        ind = (H <= threshold)
        return ind




class DropCluster2d(nn.Module):
    def __init__(self, p=0.2):
        super(DropCluster2d, self).__init__()
        self.register_buffer('p', torch.tensor(p))
        self.register_buffer('mask', torch.empty(0))
        self.register_buffer('channel_mask', torch.empty(0))

        self.cs = []
        self.activated = False
        self.prepare = False

    def extra_repr(self):
        return 'drop_prob=%.2f' % tuple([self.p.item()])

    def _activated(self):
        self.activated = True

    def _prepare(self):
        self.prepare = True

    def _mc_sampling(self):
        self.MC_testing = True

    def _cluster(self, input):
        bs, self.ch, self.feat0, self.feat1 = input.shape[0], input.shape[1], input.shape[2], input.shape[3]
        self.mask = torch.zeros_like(input[0,::]).unsqueeze(0)
        input = input.numpy()

        # Boolean roch tensor indicating unstructured channels (threshold = 0.25, somewhat arbitrary)
        self.ind = self.SpatialHopkins(input)
        self.channel_mask = self._channel_mask

        # Number of clusters per channel
        self.k = [0.] * self.ch

        # Connectivity graph
        _knn_graph = grid_to_graph(n_x=self.feat0, n_y=self.feat1, return_as=np.ndarray)

        for i in range(self.ch):
            # If not unstructured, then proceed
            if not self.ind[i]:
                # Determine the optimal number of clusters
                self.k[i] = GAP_STATISTICS(input[:,i,:,:],np.arange(2, (self.feat0 * self.feat1).sqrt()))

                # Clustering algorithm
                clstr = FeatureAgglomeration(linkage='ward', connectivity=_knn_graph, n_clusters=self.k[i])
                clstr.fit(input[:,i,:,:])
                self.mask[i,::] = torch.from_numpy(clstr.labels_)
                self.cs[i].append(Counter(clstr.labels_))

        self.prepare = False

    def _binary_mask(self, s0): # s0 = batchSize
        # This version uses the same dropout mask for all samples in the batch
        bm = torch.ones((1, self.ch, self.feat0, self.feat1))
        bm *= self.channel_mask
        numel = bm.sum(-1) * s0
        for i in range(self.ch):
            div = []
            for _, v in self.cs[i].items():
                div.append(v)
            div = torch.FloatTensor(div)
            gamma = (1 - self.p) / div
            gamma *= (self.feat0 * self.feat1) / (self.feat0 * self.feat1 - div + 1)
            mask = self._expand_mask(self.mask[i,::], self.k[i]).unsqueeze(0)
            prob = torch.rand(1,len(gamma))
            for n in range(len(gamma)):
                clusters = (prob[:,n] < gamma[n])
                mask[clusters,::].fill_(0)
            bm[:,i,:,:] = mask.sum(0).unsqueeze(0)
            bm = bm.repeat(s0)

        return bm, numel, torch.sum(bm, dim=[3,2])

    def _binary_mask_v0(self, s0): # s0 = batchSize
        # This versions uses separate dropout masks for each sample in the batch
        bm = torch.ones((s0, self.ch, self.feat0, self.feat1))
        bm *= self.channel_mask
        numel = bm.sum(-1)
        for i in range(self.ch):
            div = []
            for _, v in self.cs[i].items():
                div.append(v)
            div = torch.FloatTensor(div)
            gamma = (1 - self.p) / div
            gamma *= (self.feat0 * self.feat1) / (self.feat0 * self.feat1 - div + 1)
            mask = self._expand_mask(self.mask[i,::], self.k[i]).unsqueeze(0).repeat(s0)
            prob = torch.rand(s0,len(gamma))
            for n in range(len(gamma)):
                clusters = (prob[:,n] < gamma[n])
                mask[clusters,::].fill_(0)
            bm[:,i,:,:] = mask.sum(0).unsqueeze(0)

        return bm, numel, torch.sum(bm, dim=[3,2])

    def _expand_mask(self, mask, num_clusters):
        expanded = torch.empty_like(mask).unsqueeze(0).repeat(num_clusters)
        lb = LabelBinarizer()
        lb.fit(np.asarray(mask))
        for i in range(num_clusters):
            expanded[i,::] = torch.from_numpy(lb.transform([i]))
        return expanded

    @property
    def _channel_mask(self):
        mask = torch.ones_like(self.mask)
        return mask[self.ind,:,:].fill_(0).unsqueeze(0)

    def forward(self, x):
        if self.prepare:
            self._cluster(x)
            return x
        if not self.activated: return x
        if not self.training and not self.MC_testing: return x * self.channel_mask.repeat(x.shape[0])
        elif self.training and not self.MC_testing:
            bm, numel, summ = self._binary_mask(x.shape[0])
        elif self.MC_testing:
            bm, numel, summ = self._binary_mask_v0(x.shape[0])

        return x * bm * numel / summ

    def SpatialHopkins(self, input):
        H, n = SpatialHopkinsStatistic(input)
        # Reference: To Cluster, or Not to Cluster:An Analysis of Clusterability Methods (p.9)
        alpha = 0.05
        threshold = torch.from_numpy(beta.ppf(1-alpha, n[0], n[1]))#, dtype=torch.float)
        H = H.mean(dim=0)
        ind = (H <= threshold)
        return ind


class DropCluster3d(nn.Module):
    def __init__(self, p=0.2):
        super(DropCluster3d, self).__init__()
        self.register_buffer('p', torch.tensor(p))
        self.register_buffer('mask', torch.empty(0))
        self.register_buffer('channel_mask', torch.empty(0))
        self.cs = []
        self.activated = False
        self.prepare = False

    def extra_repr(self):
        return 'drop_prob=%.2f' % tuple([self.p.item()])

    def _activated(self):
        self.activated = True

    def _prepare(self):
        self.prepare = True

    def _mc_sampling(self):
        self.MC_testing = True

    def _cluster(self, input):
        bs, self.ch, self.feat0, self.feat1, self.feat2 = input.shape[0], input.shape[1], input.shape[2], input.shape[3], input.shape[4]
        self.mask = torch.zeros_like(input[0,::]).unsqueeze(0)
        input = input.numpy()

        # Boolean roch tensor indicating unstructured channels (threshold = 0.25, somewhat arbitrary)
        self.ind = self.SpatialHopkins(input)
        self.channel_mask = self._channel_mask

        # Number of clusters per channel
        self.k = [0.] * self.ch

        # Connectivity graph
        _knn_graph = grid_to_graph(n_x=self.feat0, n_y=self.feat1, n_z=self.feat2, return_as=np.ndarray)

        for i in range(self.ch):
            # If not unstructured, then proceed
            if not self.ind[i]:
                # Determine the optimal number of clusters
                self.k[i] = GAP_STATISTICS(input[:, i, :, :, :],np.arange(2, (self.feat0 * self.feat1 * self.feat2).sqrt()))

                # Clustering algorithm
                clstr = FeatureAgglomeration(linkage='ward', connectivity=_knn_graph, n_clusters=self.k[i])
                clstr.fit(input[:,i, :, :, :])
                self.mask[i, ::] = torch.from_numpy(clstr.labels_)
                self.cs[i].append(Counter(clstr.labels_))

        self.prepare = False


    def _binary_mask(self, s0): # s0 = batchSize
        # This version uses the same dropout mask for all samples in the batch
        bm = torch.ones((1, self.ch, self.feat0, self.feat1, self.feat2))
        bm *= self.channel_mask
        numel = bm.sum(-1) * s0
        for i in range(self.ch):
            div = []
            for _, v in self.cs[i].items(): div.append(v)
            div = torch.FloatTensor(div)
            gamma = (1 - self.p) / div
            gamma *= (self.feat0 * self.feat1 * self.feat2) / (self.feat0 * self.feat1 * self.feat2 - div + 1)
            mask = self._expand_mask(self.mask[i,::], self.k[i]).unsqueeze(0)
            prob = torch.rand(1,len(gamma))
            for n in range(len(gamma)):
                clusters = (prob[:,n] < gamma[n])
                mask[clusters,::].fill_(0)
            bm[:,i,:,:,:] = mask.sum(0).unsqueeze(0)
            bm = bm.repeat(s0)

        return bm, numel, torch.sum(bm, dim=[4,3,2])


    def _binary_mask_v0(self, s0): # s0 = batchSize
        # This versions uses separate dropout masks for each sample in the batch
        bm = torch.ones((s0, self.ch, self.feat0, self.feat1))
        bm *= self.channel_mask
        numel = bm.sum(-1)
        for i in range(self.ch):
            div = []
            for _, v in self.cs[i].items(): div.append(v)
            div = torch.FloatTensor(div)
            gamma = (1 - self.p) / div
            gamma *= (self.feat0 * self.feat1 * self.feat2) / (self.feat0 * self.feat1 * self.feat2 - div + 1)
            mask = self._expand_mask(self.mask[i,::], self.k[i]).unsqueeze(0).repeat(s0)
            prob = torch.rand(s0,len(gamma))
            for n in range(len(gamma)):
                clusters = (prob[:,n] < gamma[n])
                mask[clusters,::].fill_(0)
            bm[:,i,:,:,:] = mask.sum(0).unsqueeze(0)

        return bm, numel, torch.sum(bm, dim=[4,3,2])

    def _expand_mask(self, mask, num_clusters):
        expanded = torch.empty_like(mask).unsqueeze(0).repeat(num_clusters)
        lb = LabelBinarizer()
        lb.fit(mask.numpy())
        for i in range(num_clusters):
            expanded[i,::] = torch.from_numpy(lb.transform([i]))
        return expanded


    @property
    def _channel_mask(self):
        mask = torch.ones_like(self.mask)
        return mask[self.ind,::].fill_(0).unsqueeze(0)


    def forward(self, x):
        if self.prepare:
            self._cluster(x)
            return x
        if not self.activated: return x
        if not self.training and not self.MC_testing: return x * self.channel_mask.repeat(x.shape[0])
        elif self.training and not self.MC_testing:
            bm, numel, summ = self._binary_mask(x.shape[0])
        elif self.MC_testing:
            bm, numel, summ = self._binary_mask_v0(x.shape[0])

        return x * bm * numel / summ

    def SpatialHopkins(self, input):
        H, n = SpatialHopkinsStatistic(input)
        # Reference: To Cluster, or Not to Cluster:An Analysis of Clusterability Methods (p.9)
        alpha = 0.05
        threshold = torch.from_numpy(beta.ppf(1-alpha, n[0], n[1], n[2]))#, dtype=torch.float)
        H = H.mean(dim=0)
        ind = (H <= threshold)
        return ind



def SpatialHopkinsStatistic(input):
    if input.dim()==3:
        return SpatialHopkinsStatistic1d(input)
    elif input.dim()==4:
        return  SpatialHopkinsStatistic2d(input)
    elif input.dim()==5:
        return SpatialHopkinsStatistic3d(input)


def SpatialHopkinsStatistic1d(x):
    '''
    Statistic used to assess cluster tendency. Will be used in the DropCluster implementation.
    args:
        x: convolutional weight matrix of size [f, d, n]
        H: spatial Hopkins statistic matrix of size [f, d, 1]

    [1] Validating Clusters using the Hopkins Statistic from IEEE 2004
    '''
    f, d, n = x.shape
    x = np.asarray(x)
    m = int(0.1 * n)

    rand_X = sample(range(1, n-1, 1), m)

    ujd = np.zeros([f, d, 1])
    wjd = np.zeros([f, d, 1])
    u_pts = np.random.uniform(low=1, high=n-1, size=(f, d, m))
    for j in range(0, m):
        u_dist = (L2dist(x[:, :, u_pts[j]], x[:, :, u_pts[j] - 1]) + L2dist(x[:, :, u_pts[j]], x[:, :, u_pts[j] + 1])) / 2
        ujd += u_dist
        w_dist = (L2dist(x[:, :, rand_X[j]], x[:, :, rand_X[j] - 1]) + L2dist(x[:, :, rand_X[j]], x[:, :, rand_X[j] + 1])) / 2
        wjd += w_dist

    H = np.sum(ujd, axis=-1) / (np.sum(ujd, axis=-1) + np.sum(wjd, axis=-1))
    return torch.from_numpy(H), m

def SpatialHopkinsStatistic2d(x):
    '''
    Statistic used to assess cluster tendency. Will be used in the DropCluster implementation.
    args:
        x: convolutional output map of size [f, d, n0, n1]
        H: spatial Hopkins statistic matrix of size [f, d, 1]

    [1] Validating Clusters using the Hopkins Statistic from IEEE 2004
    '''
    f, d, n0, n1 = x.shape
    x = np.asarray(x)
    m0 = int(0.1 * n0)
    m1 = int(0.1 * n1)

    rand_X = sample(range(1, n0-1, 1), m0)
    rand_Y = sample(range(1, n1-1, 1), m1)

    ujd = np.zeros([f, d, 1])
    wjd = np.zeros([f, d, 1])
    u_pts_x = np.random.uniform(low=1, high=n0-1, size=(f, d, m0))
    u_pts_y = np.random.uniform(low=1, high=n1-1, size=(f, d, m1))
    for i in range(0, m0):
        for j in range(0, m1):
            u_dist = (L2dist(x[:, :, u_pts_x[i], u_pts_y[j]], x[:, :, u_pts_x[i] - 1, u_pts_y[j] - 1]) +
                      L2dist(x[:, :, u_pts_x[i], u_pts_y[j]], x[:, :, u_pts_x[i] - 1, u_pts_y[j] + 1]) +
                      L2dist(x[:, :, u_pts_x[i], u_pts_y[j]], x[:, :, u_pts_x[i] + 1, u_pts_y[j] - 1]) +
                      L2dist(x[:, :, u_pts_x[i], u_pts_y[j]], x[:, :, u_pts_x[i] + 1, u_pts_y[j] + 1]) +
                      L2dist(x[:, :, u_pts_x[i], u_pts_y[j]], x[:, :, u_pts_x[i]    , u_pts_y[j] + 1]) +
                      L2dist(x[:, :, u_pts_x[i], u_pts_y[j]], x[:, :, u_pts_x[i]    , u_pts_y[j] - 1]) +
                      L2dist(x[:, :, u_pts_x[i], u_pts_y[j]], x[:, :, u_pts_x[i] + 1, u_pts_y[j]    ]) +
                      L2dist(x[:, :, u_pts_x[i], u_pts_y[j]], x[:, :, u_pts_x[i] - 1, u_pts_y[j]    ])) / 8
            ujd += u_dist
            w_dist = (L2dist(x[:, :, rand_X[i], rand_Y[j]], x[:, :, rand_X[i] - 1, rand_Y[j] - 1]) +
                      L2dist(x[:, :, rand_X[i], rand_Y[j]], x[:, :, rand_X[i] - 1, rand_Y[j] + 1]) +
                      L2dist(x[:, :, rand_X[i], rand_Y[j]], x[:, :, rand_X[i] + 1, rand_Y[j] - 1]) +
                      L2dist(x[:, :, rand_X[i], rand_Y[j]], x[:, :, rand_X[i] + 1, rand_Y[j] + 1]) +
                      L2dist(x[:, :, rand_X[i], rand_Y[j]], x[:, :, rand_X[i]    , rand_Y[j] + 1]) +
                      L2dist(x[:, :, rand_X[i], rand_Y[j]], x[:, :, rand_X[i]    , rand_Y[j] - 1]) +
                      L2dist(x[:, :, rand_X[i], rand_Y[j]], x[:, :, rand_X[i] + 1, rand_Y[j]    ]) +
                      L2dist(x[:, :, rand_X[i], rand_Y[j]], x[:, :, rand_X[i] - 1, rand_Y[j]    ])) / 8
            wjd += w_dist

    H = np.sum(ujd, axis=-1) / (np.sum(ujd, axis=-1) + np.sum(wjd, axis=-1))
    return torch.from_numpy(H), (m0, m1)

def SpatialHopkinsStatistic3d(x):
    '''
    Statistic used to assess cluster tendency. Will be used in the DropCluster implementation.
    args:
        x: convolutional output map of size [f, d, n0, n1, n2]
        H: spatial Hopkins statistic matrix of size [f, d, 1]

    [1] Validating Clusters using the Hopkins Statistic from IEEE 2004
    '''
    f, d, n0, n1, n2 = x.shape
    x = np.asarray(x)
    m0 = int(0.1 * n0)
    m1 = int(0.1 * n1)
    m2 = int(0.1 * n2)

    rand_X = sample(range(1, n0-1, 1), m0)
    rand_Y = sample(range(1, n1-1, 1), m1)
    rand_Z = sample(range(1, n2-1, 1), m2)

    ujd = np.zeros([f, d, 1])
    wjd = np.zeros([f, d, 1])
    u_pts_x = np.random.uniform(low=1, high=n0-1, size=(f, d, m0))
    u_pts_y = np.random.uniform(low=1, high=n1-1, size=(f, d, m1))
    u_pts_z = np.random.uniform(low=1, high=n1-1, size=(f, d, m2))
    for i in range(0, m0):
        for j in range(0, m1):
            for k in range(0, m2):
                u_dist = (L2dist(x[:, :, u_pts_x[i], u_pts_y[j], u_pts_z[k]], x[:, :, u_pts_x[i] - 1, u_pts_y[j] - 1, u_pts_z[k]]) +
                          L2dist(x[:, :, u_pts_x[i], u_pts_y[j], u_pts_z[k]], x[:, :, u_pts_x[i] - 1, u_pts_y[j] + 1, u_pts_z[k]]) +
                          L2dist(x[:, :, u_pts_x[i], u_pts_y[j], u_pts_z[k]], x[:, :, u_pts_x[i] + 1, u_pts_y[j] - 1, u_pts_z[k]]) +
                          L2dist(x[:, :, u_pts_x[i], u_pts_y[j], u_pts_z[k]], x[:, :, u_pts_x[i] + 1, u_pts_y[j] + 1, u_pts_z[k]]) +
                          L2dist(x[:, :, u_pts_x[i], u_pts_y[j], u_pts_z[k]], x[:, :, u_pts_x[i]    , u_pts_y[j] + 1, u_pts_z[k]]) +
                          L2dist(x[:, :, u_pts_x[i], u_pts_y[j], u_pts_z[k]], x[:, :, u_pts_x[i]    , u_pts_y[j] - 1, u_pts_z[k]]) +
                          L2dist(x[:, :, u_pts_x[i], u_pts_y[j], u_pts_z[k]], x[:, :, u_pts_x[i] + 1, u_pts_y[j]    , u_pts_z[k]]) +
                          L2dist(x[:, :, u_pts_x[i], u_pts_y[j], u_pts_z[k]], x[:, :, u_pts_x[i] - 1, u_pts_y[j]    , u_pts_z[k]]) +
                          
                          L2dist(x[:, :, u_pts_x[i], u_pts_y[j], u_pts_z[k]], x[:, :, u_pts_x[i] - 1, u_pts_y[j] - 1, u_pts_z[k] + 1]) +
                          L2dist(x[:, :, u_pts_x[i], u_pts_y[j], u_pts_z[k]], x[:, :, u_pts_x[i] - 1, u_pts_y[j] + 1, u_pts_z[k] + 1]) +
                          L2dist(x[:, :, u_pts_x[i], u_pts_y[j], u_pts_z[k]], x[:, :, u_pts_x[i] + 1, u_pts_y[j] - 1, u_pts_z[k] + 1]) +
                          L2dist(x[:, :, u_pts_x[i], u_pts_y[j], u_pts_z[k]], x[:, :, u_pts_x[i] + 1, u_pts_y[j] + 1, u_pts_z[k] + 1]) +
                          L2dist(x[:, :, u_pts_x[i], u_pts_y[j], u_pts_z[k]], x[:, :, u_pts_x[i]    , u_pts_y[j] + 1, u_pts_z[k] + 1]) +
                          L2dist(x[:, :, u_pts_x[i], u_pts_y[j], u_pts_z[k]], x[:, :, u_pts_x[i]    , u_pts_y[j] - 1, u_pts_z[k] + 1]) +
                          L2dist(x[:, :, u_pts_x[i], u_pts_y[j], u_pts_z[k]], x[:, :, u_pts_x[i] + 1, u_pts_y[j]    , u_pts_z[k] + 1]) +
                          L2dist(x[:, :, u_pts_x[i], u_pts_y[j], u_pts_z[k]], x[:, :, u_pts_x[i] - 1, u_pts_y[j]    , u_pts_z[k] + 1]) +
                          
                          L2dist(x[:, :, u_pts_x[i], u_pts_y[j], u_pts_z[k]], x[:, :, u_pts_x[i] - 1, u_pts_y[j] - 1, u_pts_z[k] - 1]) +
                          L2dist(x[:, :, u_pts_x[i], u_pts_y[j], u_pts_z[k]], x[:, :, u_pts_x[i] - 1, u_pts_y[j] + 1, u_pts_z[k] - 1]) +
                          L2dist(x[:, :, u_pts_x[i], u_pts_y[j], u_pts_z[k]], x[:, :, u_pts_x[i] + 1, u_pts_y[j] - 1, u_pts_z[k] - 1]) +
                          L2dist(x[:, :, u_pts_x[i], u_pts_y[j], u_pts_z[k]], x[:, :, u_pts_x[i] + 1, u_pts_y[j] + 1, u_pts_z[k] - 1]) +
                          L2dist(x[:, :, u_pts_x[i], u_pts_y[j], u_pts_z[k]], x[:, :, u_pts_x[i]    , u_pts_y[j] + 1, u_pts_z[k] - 1]) +
                          L2dist(x[:, :, u_pts_x[i], u_pts_y[j], u_pts_z[k]], x[:, :, u_pts_x[i]    , u_pts_y[j] - 1, u_pts_z[k] - 1]) +
                          L2dist(x[:, :, u_pts_x[i], u_pts_y[j], u_pts_z[k]], x[:, :, u_pts_x[i] + 1, u_pts_y[j]    , u_pts_z[k] - 1]) +
                          L2dist(x[:, :, u_pts_x[i], u_pts_y[j], u_pts_z[k]], x[:, :, u_pts_x[i] - 1, u_pts_y[j]    , u_pts_z[k] - 1])) / 24
                ujd += u_dist
                w_dist = (L2dist(x[:, :, rand_X[i], rand_Y[j], rand_Z[k]], x[:, :, rand_X[i] - 1, rand_Y[j] - 1, rand_Z[k]]) +
                          L2dist(x[:, :, rand_X[i], rand_Y[j], rand_Z[k]], x[:, :, rand_X[i] - 1, rand_Y[j] + 1, rand_Z[k]]) +
                          L2dist(x[:, :, rand_X[i], rand_Y[j], rand_Z[k]], x[:, :, rand_X[i] + 1, rand_Y[j] - 1, rand_Z[k]]) +
                          L2dist(x[:, :, rand_X[i], rand_Y[j], rand_Z[k]], x[:, :, rand_X[i] + 1, rand_Y[j] + 1, rand_Z[k]]) +
                          L2dist(x[:, :, rand_X[i], rand_Y[j], rand_Z[k]], x[:, :, rand_X[i]    , rand_Y[j] + 1, rand_Z[k]]) +
                          L2dist(x[:, :, rand_X[i], rand_Y[j], rand_Z[k]], x[:, :, rand_X[i]    , rand_Y[j] - 1, rand_Z[k]]) +
                          L2dist(x[:, :, rand_X[i], rand_Y[j], rand_Z[k]], x[:, :, rand_X[i] + 1, rand_Y[j]    , rand_Z[k]]) +
                          L2dist(x[:, :, rand_X[i], rand_Y[j], rand_Z[k]], x[:, :, rand_X[i] - 1, rand_Y[j]    , rand_Z[k]]) +
                          
                          L2dist(x[:, :, rand_X[i], rand_Y[j], rand_Z[k]], x[:, :, rand_X[i] - 1, rand_Y[j] - 1, rand_Z[k] + 1]) +
                          L2dist(x[:, :, rand_X[i], rand_Y[j], rand_Z[k]], x[:, :, rand_X[i] - 1, rand_Y[j] + 1, rand_Z[k] + 1]) +
                          L2dist(x[:, :, rand_X[i], rand_Y[j], rand_Z[k]], x[:, :, rand_X[i] + 1, rand_Y[j] - 1, rand_Z[k] + 1]) +
                          L2dist(x[:, :, rand_X[i], rand_Y[j], rand_Z[k]], x[:, :, rand_X[i] + 1, rand_Y[j] + 1, rand_Z[k] + 1]) +
                          L2dist(x[:, :, rand_X[i], rand_Y[j], rand_Z[k]], x[:, :, rand_X[i]    , rand_Y[j] + 1, rand_Z[k] + 1]) +
                          L2dist(x[:, :, rand_X[i], rand_Y[j], rand_Z[k]], x[:, :, rand_X[i]    , rand_Y[j] - 1, rand_Z[k] + 1]) +
                          L2dist(x[:, :, rand_X[i], rand_Y[j], rand_Z[k]], x[:, :, rand_X[i] + 1, rand_Y[j]    , rand_Z[k] + 1]) +
                          L2dist(x[:, :, rand_X[i], rand_Y[j], rand_Z[k]], x[:, :, rand_X[i] - 1, rand_Y[j]    , rand_Z[k] + 1]) +
                          
                          L2dist(x[:, :, rand_X[i], rand_Y[j], rand_Z[k]], x[:, :, rand_X[i] - 1, rand_Y[j] - 1, rand_Z[k] - 1]) +
                          L2dist(x[:, :, rand_X[i], rand_Y[j], rand_Z[k]], x[:, :, rand_X[i] - 1, rand_Y[j] + 1, rand_Z[k] - 1]) +
                          L2dist(x[:, :, rand_X[i], rand_Y[j], rand_Z[k]], x[:, :, rand_X[i] + 1, rand_Y[j] - 1, rand_Z[k] - 1]) +
                          L2dist(x[:, :, rand_X[i], rand_Y[j], rand_Z[k]], x[:, :, rand_X[i] + 1, rand_Y[j] + 1, rand_Z[k] - 1]) +
                          L2dist(x[:, :, rand_X[i], rand_Y[j], rand_Z[k]], x[:, :, rand_X[i]    , rand_Y[j] + 1, rand_Z[k] - 1]) +
                          L2dist(x[:, :, rand_X[i], rand_Y[j], rand_Z[k]], x[:, :, rand_X[i]    , rand_Y[j] - 1, rand_Z[k] - 1]) +
                          L2dist(x[:, :, rand_X[i], rand_Y[j], rand_Z[k]], x[:, :, rand_X[i] + 1, rand_Y[j]    , rand_Z[k] - 1]) +
                          L2dist(x[:, :, rand_X[i], rand_Y[j], rand_Z[k]], x[:, :, rand_X[i] - 1, rand_Y[j]    , rand_Z[k] - 1])) / 24
                wjd += w_dist

    H = np.sum(ujd, axis=-1) / (np.sum(ujd, axis=-1) + np.sum(wjd, axis=-1))
    return torch.from_numpy(H), (m0, m1, m2)
