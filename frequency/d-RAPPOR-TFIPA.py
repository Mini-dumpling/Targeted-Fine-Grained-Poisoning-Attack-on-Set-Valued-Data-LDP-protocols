

import numpy as np
import random
import math
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
import csv


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

    for x in domain:
        x_count = count_dict.get(x, 0)
        rs = (x_count - n * q) / (n * (p - q))
        es_data.append(rs)

    return es_data


def generate_data(domain: list, n: int, c: int):
    data_user = []      # 所有用户数据
    for i in range(n):
        x_raw = random.sample(domain, c)
        data_user.append(x_raw)
    return data_user


def calculate_mse(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("两个列表的长度不相等")
    squared_errors = [(x - y) ** 2 for x, y in zip(list1, list2)]
    mse = sum(squared_errors) / len(list1)
    return mse


def calculate_mae(A, B):
    if len(A) != len(B):
        return None  # 如果两个列表长度不同，则返回None
    n = len(A)
    sum_abs_diff = 0.0  # 初始化绝对值差的总和
    for i in range(n):
        sum_abs_diff += abs(A[i] - B[i])  # 计算绝对值差并累加到总和中
    mae = sum_abs_diff / n  # 计算平均绝对值差
    return mae


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


def frequency_true(domain: list, data_user: list):
    num = len(data_user)
    d = len(domain)
    result = []
    for i in range(d):
        count = 0
        for j in range(num):
            if domain[i] in data_user[j]:
                count += 1
        result.append(count/num)
    return result


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


# 二维数组每列求均值
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


# 列表每个元组保留三位小数
def round_list_values(lst):
    rounded_values = [round(value, 3) for value in lst]
    return rounded_values


def calculate_column_variance(matrix):
    arr = np.array(matrix)
    variances = np.var(arr, axis=0)
    return variances.tolist()


def Adjust_value(epsilon, r_item, r_true_fre, a):
    e_ep = np.exp(epsilon / (2 * c))
    p = e_ep / (1 + e_ep)
    q = 1 - p

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






