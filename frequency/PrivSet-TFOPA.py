

import numpy as np
import numpy.random as r
from scipy.special import comb
import random
import matplotlib.pyplot as plt
import math
import gurobipy as gp
from gurobipy import GRB
import csv


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

    print("r_solve1:", r_solve1)
    r_solve4 = []
    for i in range(r):
        r_solve4.append(r_solve2[i] - r_solve3[i])
    print("r_solve4:", r_solve4)

    # 用于近似严格不等式的小正数（默认 1e-6）
    epsilon = 1e-6

    # 创建模型
    model = gp.Model("Find_Minimal_u")

    # 添加变量 u（正整数）
    u = model.addVar(vtype=GRB.INTEGER, lb=1, name="u")

    # 设置目标：最小化 u
    model.setObjective(u, GRB.MINIMIZE)

    # 添加约束：0 < a[i] * u - b[i] < u 对每个 i
    for i in range(r):
        a = r_solve1[i]
        b = r_solve4[i]

        # 约束 1: a[i] * u - b[i] > 0 转换为 a[i] * u - b[i] >= epsilon
        model.addConstr(a * u + b >= epsilon, name=f"ineq1_{i}")

        # 约束 2: a[i] * u - b[i] < u 转换为 a[i] * u - b[i] <= u - epsilon
        model.addConstr(a * u + b <= u - epsilon, name=f"ineq2_{i}")

    # 求解模型
    model.optimize()

    # 检查是否有解
    if model.status == GRB.OPTIMAL:
        result.append(int(round(u.X)))  # 返回最小的 u（四舍五入确保整数）

    for i in range(r):
        tt = round(r_solve1[i] * result[0] + r_solve4[i])
        result.append(tt)

    print("tt:", result)
    return result


def find_min_positive(numbers):
    positive_numbers = [num for num in numbers if num > 0]
    if not positive_numbers:
        return -1
    return min(positive_numbers)


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


def calculate_mse(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("两个列表的长度不相等")
    squared_errors = [(x - y) ** 2 for x, y in zip(list1, list2)]
    mse = sum(squared_errors) / len(list1)
    return mse


def calculate_column_averages(matrix):
    row_count = len(matrix)
    if row_count == 0:
        return []
    col_count = len(matrix[0])
    column_averages = [0] * col_count
    for row in matrix:
        for i, value in enumerate(row):
            column_averages[i] += value
    for i in range(col_count):
        column_averages[i] /= row_count
    return column_averages


def round_list_values(lst):
    rounded_values = [round(value, 3) for value in lst]
    return rounded_values


def calculate_column_variance(matrix):
    arr = np.array(matrix)
    variances = np.var(arr, axis=0)
    return variances.tolist()


def Adjust_value(r_item, r_true_fre, a, d, c, ep, k):
    privset_server = PrivSet_SERVER(d, c, ep, k)
    p = privset_server.trate
    q = privset_server.frate

    r = len(r_item)
    D = [0 for col in range(r)]

    for i in range(r):
        D[i] = (r_true_fre[i]*p*(1-p) + (1-r_true_fre[i])*q*(1-q)) / (n * (p-q)**2)

    result = []
    for i in range(r):
        result.append(math.sqrt(D[i] / (10 * (1 - a))))

    return result


def trur_frequency(r_item, domain, data):
    data_list = []
    for x in data:
        data_list.extend(x)

    d = len(domain)
    temp = []
    for i in range(d):
        temp.append(data_list.count(domain[i]))

    num = len(data)
    result = []
    for i in range(d):
        result.append(temp[i]/num)

    item_fre = []
    for x in r_item:
        inx = domain.index(x)
        item_fre.append(result[inx])

    return item_fre


# 读入csv数据
def read_csv_to_int_lists(file_path):
    big_list = []
    with open(file_path, mode='r', encoding='utf-8-sig') as file:  # utf-8-sig 处理可能的BOM头
        csv_reader = csv.reader(file)
        for row in csv_reader:
            # 过滤空字符串并转换为整数
            int_list = [int(x) for x in row if x.strip()]  # 关键修改：跳过空值
            big_list.append(int_list)
    return big_list


# 截断和填充
def process_sublists(big_list, c, data_range=(0, 1656)):
    min_val, max_val = data_range
    new_big_list = []

    for sublist in big_list:
        # 去重（确保小列表内元素唯一）
        unique_elements = list(set(sublist))

        # 情况1：长度超过10，随机保留10个
        if len(unique_elements) > c:
            new_sublist = random.sample(unique_elements, c)

        # 情况2：长度不足10，补充随机数
        else:
            # 可用的补充数字 = 数据域范围 - 小列表已有数字
            available_numbers = set(range(min_val, max_val + 1)) - set(unique_elements)
            needed = c - len(unique_elements)

            # 随机选择补充数字
            if len(available_numbers) >= needed:
                supplements = random.sample(available_numbers, needed)
            else:
                # 如果数据域不足以补充（理论上不会发生，因数据域有1657个数）
                supplements = list(available_numbers)

            new_sublist = unique_elements + supplements

        new_big_list.append(new_sublist)

    return new_big_list

























