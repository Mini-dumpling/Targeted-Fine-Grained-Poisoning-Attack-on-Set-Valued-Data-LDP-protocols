

import numpy as np
import random
import math
import xxhash
import matplotlib.pyplot as plt


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


def user(epsilon, domain, data_lists):
    data = []
    for data_list in data_lists:
        wu = Wheel_USER(epsilon, domain, data_list)
        wu.run()
        data.append(wu.get_per_data())
    return data


def server(epsilon, domain, data, c):
    ws = Wheel_SERVER(epsilon, domain, data, c)
    ws.estimate()
    return ws.get_es_data()


class Wheel_USER(object):
    def __init__(self, epsilon: float, domain: list, data_list: list):
        super(Wheel_USER, self).__init__()
        # 隐私预算
        self.epsilon = epsilon
        # 原始数据定义域
        self.domain = domain
        # 用户的原始数据
        self.data_list = data_list
        # 用户手中数据的数目
        self.c = len(data_list)
        # 扰动数据
        self.per_data = 0

    # N是用户数量
    # c是数据域大小,也就是用户手中数据的条数
    # 扰动过程, X为用户的真实数据, 用户输入值的域，epsilon
    def run(self):
        # 为了直接将之前的代码拿过来用，特意设置下列值
        epsilon = self.epsilon
        c = self.c
        X = [self.data_list]
        N = 1
        # 手动抽取用户所用的hash种子
        seed = random.randint(0, 100000)

        max_int_32 = (1 << 32) - 1
        Y = [0 for col in range(N)]
        s = math.exp(epsilon)
        temp_p = 1 / (2 * c - 1 + c * s)
        omega = c * temp_p * s + (1 - c * temp_p)
        for i in range(N):
            V = [0 for col in range(c)]
            # hash
            for j in range(c):
                V[j] = xxhash.xxh32_intdigest(str(X[i][j]), seed=seed) / max_int_32
            # 区间合并的准备工作
            # 详见论文算法3
            bSize = math.ceil(1 / temp_p)
            lef = [0 for col in range(bSize)]
            rig = [0 for col in range(bSize)]
            for b in range(bSize):
                lef[b] = min((b + 1) * temp_p, 1.0)
                rig[b] = b * temp_p
            # 算法3 第6行开始
            for v in V:
                temp_b = math.ceil(v / temp_p) - 1
                lef[temp_b] = min(v, lef[temp_b])
                if temp_b < math.ceil(1 / temp_p) - 1:
                    rig[temp_b + 1] = max(v + temp_p, rig[temp_b + 1])
                else:
                    rig[0] = max(v + temp_p - 1, rig[0])
            temp_rig0 = rig[0]
            for b in range(bSize - 1):
                lef[b] = max(lef[b], rig[b])
                rig[b] = rig[b + 1]
            lef[bSize - 1] = max(lef[bSize - 1], rig[bSize - 1])
            rig[bSize - 1] = temp_rig0 + 1.0
            # 算法3 21行结束
            # ll为总长度
            ll = 0.0
            for b in range(bSize):
                ll = ll + rig[b] - lef[b]
            r = np.random.random_sample()
            a = 0.0
            for b in range(bSize):
                a = a + s * (rig[b] - lef[b]) / omega
                if a > r:
                    z = rig[b] - (a - r) * omega / s
                    break
                a = a + (omega - ll * s) * (lef[(b + 1) % round(bSize)] + math.floor((b + 1) * temp_p) - rig[b]) / (
                        (1 - ll) * omega)
                if a > r:
                    z = lef[(b + 1) % bSize] - (a - r) * (1 - ll) * omega / (omega - ll * s)
                    break
            z = z % 1.0
            Y[i] = z
        self.per_data = [seed, Y[0]]

    def get_per_data(self):
        return self.per_data


class Wheel_SERVER(object):
    def __init__(self, epsilon: float, domain: list, per_datalist: list, c: int):
        super(Wheel_SERVER, self).__init__()
        # 隐私预算
        self.epsilon = epsilon
        # 原始数据定义域
        self.domain = domain
        # 用户数量
        self.n = len(per_datalist)
        # 用户手中数据数目
        self.c = c
        # 频率估计结果
        self.es_data = []

        # 将用户扰动数据和用户种子拆分
        # 用户扰动数据集合
        self.per_datalist = []
        self.seed = []
        for x in per_datalist:
            self.seed.append(x[0])
            self.per_datalist.append(x[1])

    # Y是所有用户的扰动数据，N是行，c是列，D是原始数据域
    def estimate(self):
        Y = self.per_datalist
        N = self.n
        c = self.c
        epsilon = self.epsilon
        D = self.domain

        max_int_32 = (1 << 32) - 1
        k = len(D)
        # Estimate_Dist = [0] * k
        Estimate_Dist = [0 for col in range(k)]
        # Estimate_Dist = np.zeros(k, dtype=float)
        s = math.exp(epsilon)
        temp_p = 1 / (2 * c - 1 + c * s)
        for i in range(N):
            z = Y[i]
            for j in range(k):
                x = D[j]
                v = xxhash.xxh32_intdigest(str(x), seed=self.seed[i]) / max_int_32
                if z - temp_p < v <= z or z - temp_p + 1 < v < 1:
                    Estimate_Dist[j] += 1
        # 矫正过程
        pt = temp_p * s / (c * temp_p * s + (1 - c * temp_p))
        pf = temp_p
        # print("pt:", pt)
        # print("pf:", pf)
        # print("N:", N)
        for i in range(k):
            Estimate_Dist[i] = 1 / N * (Estimate_Dist[i] - N * pf) / (pt - pf)
        self.es_data = Estimate_Dist

    def get_es_data(self):
        return self.es_data


def generate_data(domain: list, n: int, c: int):
    data_user = []      # 所有用户数据
    for i in range(n):
        x_raw = random.sample(domain, c)
        data_user.append(x_raw)
    return data_user


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
    u = 0           # 假用户人数
    # while flag == 0:
    #     u = u + 1
    #     flag1 = 1
    #     for i in range(r):
    #         tt = r_solve1[i] * u + r_solve2[i] - r_solve3[i]
    #         if (tt % 1 != 0) or (tt not in range(0, u + 1)):
    #             flag1 = 0
    #             break
    #     if flag1 == 1:
    #         flag = 1
    #         result.append(u)
    #         for i in range(r):
    #             tt = r_solve1[i] * u + r_solve2[i] - r_solve3[i]
    #             result.append(tt)
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
        u = math.ceil(sum(r_count)/c)
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

    # # 假用户的集合
    # fake_data = [[] for _ in range(u)]
    #
    # index = 0
    # for (k, v) in r_dict.items():
    #     for i in range(v):
    #         fake_data[(index + i) % u].append(k)
    #     index += v

    # min_count = int(min(r_count))
    # while min_count >= 0:
    #     for i in range(min_count):
    #         fake_data[i].append(r_item[r_count.index(min_count)])
    #     r_count[r_count.index(min_count)] = -1
    #     min_count = int(find_min_positive(r_count))

    for x in fake_data:
        if len(x) < c:
            temp = random.sample(remain_list, c - len(x))
            # temp = padding_list[:(c - len(x))]
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


def PDGen(target_item, u, c):
    Z = []
    for i in range(u):
        temp = random.sample(target_item, c)
        Z.append(temp)
    return Z



