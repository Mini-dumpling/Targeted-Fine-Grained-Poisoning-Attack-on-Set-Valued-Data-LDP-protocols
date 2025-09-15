

import numpy as np
import math
import xxhash
import random
import matplotlib.pyplot as plt


def set_to_single(data_user, c):
    n = len(data_user)
    data_sample = [0 for col in range(n)]
    for i in range(n):
        data_sample[i] = data_user[i][np.random.randint(c)]
    return data_sample


def OLH_Perturb_sample(data_sample, epsilon):
    g = math.ceil(math.exp(epsilon) + 1)
    # print("g:", g)
    p = 1 / 2
    n = len(data_sample)
    data_perturb = [0 for col in range(n)]
    for i in range(n):
        data_perturb[i] = xxhash.xxh32(str(data_sample[i]), seed=i).intdigest() % g
        t = np.random.random()
        if t > p:
            temp = np.random.randint(g)  # 随机返回一个[0, g-1]的一个整数
            while temp == data_perturb[i]:
                temp = np.random.randint(g)
            data_perturb[i] = temp
    return data_perturb


# k代表数据取值范围，这里为D，就不需要k了
def OLH_Aggregate_sample(data_perturb, c, epsilon, domain):
    g = math.ceil(math.exp(epsilon) + 1)
    p = 1 / 2
    q = 1 / g
    d = len(domain)
    n = len(data_perturb)
    Z = [0 for col in range(d)]
    for i in range(d):
        t_count = 0
        for j in range(n):
            temp = xxhash.xxh32(str(domain[i]), seed=j).intdigest() % g
            if temp == data_perturb[j]:
                t_count += 1
        # print(domain[i], "--", t_count)
        Z[i] = c * (t_count / n - q) / (p - q)
    return Z


def generate_data(domain: list, n: int, c: int):
    data_user = []      # 所有用户数据
    for i in range(n):
        x_raw = random.sample(domain, c)
        data_user.append(x_raw)
    return data_user


def count_occurrences(A, B):
    # 初始化结果字典
    result_dict = {}
    # 遍历一维列表A
    for element in A:
        # 初始化计数器
        count = 0
        # 遍历二维列表B中的每一行
        for row in B:
            # 检查元素是否在当前行中出现
            count += row.count(element)
        # 将结果存入字典
        result_dict[element] = count
    return result_dict


def input_attack(r_item: list, r_fre: list, n: int, count_dict: dict):
    result = []
    r = len(r_item)
    r_solve1 = []
    r_solve2 = []
    r_solve3 = []

    # r个求解式子
    for i in range(r):
        t1 = r_fre[i]
        t2 = r_fre[i] * n
        t3 = count_dict[r_item[i]]
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
    # if sum(r_count) > (u * c):
    #     u = math.ceil(sum(r_count)/c)
    if sum(r_count) > u:
        u = sum(r_count)
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


def PDGen(target_item, u):
    Z = []
    for i in range(u):
        temp = random.choice(target_item)
        Z.append(temp)
    return Z

