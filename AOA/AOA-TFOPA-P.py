
import numpy as np
import numpy.random as r
from scipy.special import comb
import random
import matplotlib.pyplot as plt
import math


class PrivSet:
    name = 'PRIVSET'
    ep = 0.0    # privacy budget epsilon
    d = 0       # domain size + maximum subset size
    c = 0       # maximum subset size
    trate = 0.0     # hit rate when true
    frate = 0.0     # hit rate when false
    normalizer = 0.0    # normalizer for proportional probabilities

    def __init__(self, d, m, ep, k=None):
        self.ep = ep
        self.d = d
        self.m = m
        self.k = k

        self.__setparams()

    def __setparams(self):
        if self.k == None:
            self.k = self.bestSubsetSize(self.d, self.m, self.ep)[0]
        interCount = comb(self.d + self.m, self.k) - comb(self.d, self.k)
        nonInterCount = comb(self.d, self.k)
        normalizer = nonInterCount + interCount * np.exp(self.ep)
        self.normalizer = normalizer

    @staticmethod
    def bestSubsetSize(d, m, ep):
        errorbounds = np.full(d+m, 0.0)
        infos = [None] * (d+m)
        for k in range(1, d):
            interCount = comb(d+m, k) - comb(d, k)
            nonInterCount = comb(d, k)
            normalizer = nonInterCount + interCount*np.exp(ep)
            trate = comb(d+m-1, k-1) * np.exp(ep)/normalizer
            frate = (comb(d-1, k-1)+(interCount * k - comb(d+m-1, k-1) * m) * np.exp(ep) / d) / normalizer
            errorbounds[k] = (trate * (1.0-trate) + (d+m-1) * frate * (1.0-frate)) / ((trate-frate) * (trate-frate))
            infos[k] = [trate, frate, errorbounds[k]]
        bestk = np.argmin(errorbounds[1:d])+1
        return [bestk]+infos[bestk]

    def randomizer(self, secrets, domain):
        pub = np.zeros(self.d, dtype=int)    # 初始化零数组
        probs = np.full(self.k+1, 0.0)              # 初始化零数组
        for inter in range(0, self.k+1):
            probs[inter] = comb(self.m, inter) * comb(self.d, self.k-inter) / self.normalizer
        probs = probs * np.exp(self.ep)
        probs[0] = probs[0] / np.exp(self.ep)

        for inter in range(1, self.k+1):
            probs[inter] += probs[inter-1]

        p = r.random(1)[0]                          # 取数组中第一个数据，即取一个随机数
        sinter = 0
        while probs[sinter] <= p:
            sinter += 1

        # 填充数据集
        # domain_pad = domain + []
        # for i in range(self.m):
        #     domain_pad.append(self.d + i)

        remain = list(set(domain)-set(secrets))
        pubset = random.sample(secrets, sinter) + random.sample(remain, self.k-sinter)
        for i in range(0, self.d):
            if i in pubset:
                pub[i] = 1
        return pub

    def get_k(self):
        return self.k


class PrivSet_SERVER:
    ep = 0.0    # privacy budget epsilon
    d = 0       # domain size + maximum subset size
    c = 0       # maximum subset size
    trate = 0.0     # hit rate when true
    frate = 0.0     # hit rate when false
    normalizer = 0.0    # normalizer for proportional probabilities

    def __init__(self, d, m, ep, k=None):
        self.ep = ep
        self.d = d
        self.m = m
        self.k = k

        self.__setparams()

    def __setparams(self):
        if self.k == None:
            self.k = self.bestSubsetSize(self.d, self.m, self.ep)[0]
        interCount = comb(self.d + self.m, self.k) - comb(self.d, self.k)
        nonInterCount = comb(self.d, self.k)
        normalizer = nonInterCount + interCount * np.exp(self.ep)
        self.normalizer = normalizer
        self.trate = comb(self.d + self.m - 1, self.k - 1) * np.exp(self.ep) / normalizer
        self.frate = (comb(self.d - 1, self.k - 1) + (interCount * self.k - comb(self.d + self.m - 1, self.k - 1) * self.m) * np.exp(self.ep) / self.d) / normalizer


    @staticmethod
    def bestSubsetSize(d, m, ep):
        errorbounds = np.full(d+m, 0.0)
        infos = [None] * (d+m)
        for k in range(1, d):
            interCount = comb(d+m, k) - comb(d, k)
            nonInterCount = comb(d, k)
            normalizer = nonInterCount + interCount*np.exp(ep)
            trate = comb(d+m-1, k-1) * np.exp(ep) / normalizer
            frate = (comb(d-1, k-1) + (interCount * k - comb(d+m-1, k-1) * m) * np.exp(ep) / d) / normalizer
            errorbounds[k] = (trate * (1.0-trate) + (d+m-1) * frate * (1.0-frate)) / ((trate-frate) * (trate-frate))
            infos[k] = [trate, frate, errorbounds[k]]
        bestk = np.argmin(errorbounds[1:d]) + 1
        return [bestk]+infos[bestk]


    def estimate(self, domain, hits):
        es_count = []

        # 在获取扰动数据中元素频率时，一定要用字典，可以大大节省运行时间
        array = np.sum(hits, axis=0)
        count_dict = dict(zip(domain, array))

        for x in domain:
            x_count = count_dict.get(x, 0)
            es_count.append(x_count)
        return es_count

    def decoder(self, domain, hits):
        # debias hits but without projecting to simplex
        es_data = []

        # 在获取扰动数据中元素频率时，一定要用字典，可以大大节省运行时间
        array = np.sum(hits, axis=0)
        count_dict = dict(zip(domain, array))

        num = len(hits)
        # print(num)

        for x in domain:
            x_count = count_dict.get(x, 0)
            fs = (x_count/num - self.frate) / (self.trate-self.frate)
            es_data.append(fs)

        return es_data


# 产生预处理后的用户数据
def generate_data(domain: list, n: int, c: int):
    data_user = []  # 所有用户数据
    for i in range(n):
        x_raw = random.sample(domain, c)
        data_user.append(x_raw)
    return data_user


def run(data: list, domain: list, m: int, ep, k):
    per_data = []   # 扰动数据
    d = len(domain)
    privset = PrivSet(d, m, ep, k)
    for x in data:
        x_per = privset.randomizer(x, domain)
        per_data.append(x_per.tolist())
    return per_data


def frequency_es(per_data:list, d:int, m:int, ep, k):
    privset_server = PrivSet_SERVER(d, m, ep, k)
    ct = privset_server.estimate(domain, per_data)
    fs = privset_server.decoder(domain, per_data)
    return ct, fs


def normalization(frequency: list, m: int):
    frequency_min = min(frequency)
    f_sum = 0
    frequency_new = []
    for f in frequency:
        f_sum = f_sum + (f - frequency_min)
    for f in frequency:
        f =((f - frequency_min) / f_sum) * m
        frequency_new.append(f)
    return frequency_new


def output_attack(ep: float, c: int, r_item: list, r_fre: list, n: int, C_dict, d, k):
    result = []
    r = len(r_item)
    r_solve1 = []
    r_solve2 = []
    r_solve3 = []

    privset_server = PrivSet_SERVER(d, c, ep, k)
    p = privset_server.trate
    q = privset_server.frate

    # r个求解式子
    for i in range(r):
        t1 = r_fre[i] * (p - q) + q
        t2 = (r_fre[i] * (p - q) + q) * n
        t3 = C_dict[r_item[i]]
        r_solve1.append(t1)
        r_solve2.append(t2)
        r_solve3.append(t3)

    flag = 0
    u = 0  # 假用户人数
    while flag == 0:
        u = u + 1
        flag1 = 1
        for i in range(r):
            tt = r_solve1[i] * u + r_solve2[i] - r_solve3[i]
            if tt < 0 or tt > u:
                flag1 = 0
                break
        if flag1 == 1:
            flag = 1
            result.append(u)
            for i in range(r):
                tt = round(r_solve1[i] * u + r_solve2[i] - r_solve3[i])
                result.append(tt)

    return result


def generate_fake_data(att_result: list, r_item: list, remain_list: list, d, c, ep, k):
    u = att_result[0]
    l = len(att_result)
    r_count = list(map(int, att_result[1:l]))  # 每个目标项目增加的次数，元素转为int类型
    r_dict = dict(zip(r_item, r_count))

    privset_server = PrivSet_SERVER(d, c, ep, k)
    m = privset_server.k
    # print("k:", m)

    # 假用户的集合
    if sum(r_count) > u:
        u = sum(r_count)
        fake_data = [[] for _ in range(u)]
        index = 0
        for (ke, v) in r_dict.items():
            for i in range(v):
                fake_data[(index + i) % u].append(ke)
            index += v

    else:
        fake_data = [[] for _ in range(u)]
        index = 0
        for (ke, v) in r_dict.items():
            for i in range(v):
                fake_data[(index + i) % u].append(ke)
            index += v

    result = []
    for x in fake_data:
        if len(x) > m:
            x = random.sample(x, m)
            result.append(x)
        elif len(x) < m:
            x = random.sample(remain_list, m-len(x))
            result.append(x)
        else:
            result.append(x)

    return result


def convert_Binary(domain: list, data_list: list):
    Binary_result = []
    for x in data_list:
        temp = [0] * len(domain)
        for i in x:
            t = domain.index(i)
            temp[t] = 1
        Binary_result.append(temp)
    return Binary_result


def split_sorted_indices(lst, k):
    # 将列表的值和原索引一起打包
    indexed_lst = [(value, idx) for idx, value in enumerate(lst)]
    # 按值从大到小排序
    sorted_lst = sorted(indexed_lst, key=lambda x: x[0], reverse=True)
    # 提取前k项的索引
    first_k_indices = [item[1] for item in sorted_lst[:k]]
    # 提取第k+1到2k项的索引
    next_k_indices = [item[1] for item in sorted_lst[k:2 * k]]
    return first_k_indices, next_k_indices


#计算归一化累积排名NCR
def calculate_ncr(true_top_k, estimated_top_k, k):
    # 计算真实 top-k 项的分数
    true_scores = {}
    for rank, item in enumerate(true_top_k, start=1):
        true_scores[item] = k - (rank - 1)

    # 计算估计 top-k 项的累积分数
    cumulative_score = 0
    for item in estimated_top_k:
        if item in true_scores:
            cumulative_score += true_scores[item]

    # 计算真实 top-k 项的总分数
    total_score = sum(true_scores.values())

    # 计算 NCR
    ncr = cumulative_score / total_score if total_score != 0 else 0
    return ncr


# 计算准确率ACC
def calculate_acc(true_top_k, estimated_top_k, k):
    # 计算交集的大小
    intersection_size = len(set(true_top_k) & set(estimated_top_k))

    # 计算 ACC
    acc = intersection_size / k
    return acc


def PDGen(target_item, u, c):
    Z = []
    for i in range(u):
        temp = random.sample(target_item, c)
        Z.append(temp)
    return Z



