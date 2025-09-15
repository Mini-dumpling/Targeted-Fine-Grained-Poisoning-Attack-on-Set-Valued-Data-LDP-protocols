

import numpy as np
import random
import math
import matplotlib.pyplot as plt


def encode(domain, data_user):
    l = list()
    d = len(domain)
    for data in data_user:
        temp = [0] * d
        for x in data:
            inx = domain.index(x)
            temp[inx] = 1
        l.append(temp)
    return l


def perturb(data_encode, domain, epsilon, c):
    d = len(domain)

    e_ep = np.exp(epsilon / (2 * c))
    p = e_ep / (1 + e_ep)
    q = 1 - p

    for data in data_encode:
        for i in range(d):
            if np.random.uniform() > p:
                if data[i] == 1:
                    data[i] = 0
                elif data[i] == 0:
                    data[i] = 1
    return data_encode


def estimate(data_perturb, domain, epsilon, c):
    array = np.sum(data_perturb, axis=0)
    count_dict = dict(zip(domain, array))

    e_ep = np.exp(epsilon / (2 * c))
    p = e_ep / (1 + e_ep)
    q = 1 - p

    n = len(data_perturb)

    es_data = []
    Ct = []

    for x in domain:
        x_count = count_dict.get(x, 0)
        rs = (x_count - n * q) / (n * (p - q))
        es_data.append(rs)
        Ct.append(x_count)

    return Ct, es_data


def data_fake_estimate(data_fake_encode, domain, epsilon, c, ct, n):
    array = np.sum(data_fake_encode, axis=0)
    count_dict = dict(zip(domain, array))

    e_ep = np.exp(epsilon / (2 * c))
    p = e_ep / (1 + e_ep)
    q = 1 - p

    N = len(data_fake_encode) + n

    es_data = []

    for x in domain:
        x_count = count_dict.get(x, 0)
        x_count_all = x_count + ct.get(x, 0)
        rs = (x_count_all - N * q) / (N * (p - q))
        es_data.append(rs)

    return es_data


def generate_data(domain: list, n: int, c: int):
    data_user = []      # 所有用户数据
    for i in range(n):
        x_raw = random.sample(domain, c)
        data_user.append(x_raw)
    return data_user


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


def output_attack(epsilon: float, c: int, r_item: list, r_fre: list, n: int, S_dict):
    result = []
    r = len(r_item)
    r_solve1 = []
    r_solve2 = []
    r_solve3 = []

    e_ep = np.exp(epsilon / (2 * c))
    p = e_ep / (1 + e_ep)
    q = 1 - p

    # r个求解式子
    for i in range(r):
        t1 = r_fre[i] * (p - q) + q
        t2 = (r_fre[i] * (p - q) + q) * n
        t3 = S_dict[r_item[i]]
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


def generate_fake_data(att_result: list, r_item: list, remain_list: list, c: int):
    u = att_result[0]
    l = len(att_result)
    r_count = list(map(int, att_result[1:l]))  # 每个目标项目增加的次数，元素转为int类型
    r_dict = dict(zip(r_item, r_count))

    # 假用户的集合
    if sum(r_count) > (u * c):
        u = math.ceil(sum(r_count) / c)
        fake_data = [[] for _ in range(u)]
        index = 0
        for (k, v) in r_dict.items():
            for i in range(v):
                fake_data[(index + i) % u].append(k)
            index += v

    else:
        fake_data = [[] for _ in range(u)]
        index = 0
        for (k, v) in r_dict.items():
            for i in range(v):
                fake_data[(index + i) % u].append(k)
            index += v

    for x in fake_data:
        if len(x) < c:
            temp = random.sample(remain_list, c - len(x))
            for j in temp:
                x.append(j)
        if len(x) > c:
            temp = random.sample(x, c)
            fake_data[fake_data.index(x)] = temp

    return fake_data


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


def PDGen(target_item, u, c, d):
    Z = []
    for i in range(u):
        temp = [0] * d
        r_item = random.sample(target_item, c)
        for x in r_item:
            temp[x] = 1
        Z.append(temp)
    return Z


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




