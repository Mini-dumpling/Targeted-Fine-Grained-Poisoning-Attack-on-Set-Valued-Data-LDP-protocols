

import numpy as np
import math
import xxhash
import random
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
import csv


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
    count = [0 for col in range(d)]
    Z = [0 for col in range(d)]

    for i in range(d):
        t_count = 0
        for j in range(n):
            temp = xxhash.xxh32(str(domain[i]), seed=j).intdigest() % g
            if temp == data_perturb[j]:
                t_count += 1
        count[i] = t_count
        # print(domain[i], "--", t_count)
        Z[i] = c * (t_count / n - q) / (p - q)
    return count, Z


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


def output_attack(epsilon: float, r_item: list, r_fre: list, n: int, S_dict, c: int):
    result = []
    r = len(r_item)
    r_solve1 = []
    r_solve2 = []
    r_solve3 = []

    g = math.ceil(math.exp(epsilon) + 1)
    p = 1 / 2
    q = 1 / g

    # r个求解式子
    for i in range(r):
        t1 = (r_fre[i] / c) * (p - q) + q
        t2 = ((r_fre[i] / c) * (p - q) + q) * n
        t3 = S_dict[r_item[i]]
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


def generate_fake_data_olh_sample(att_result: list, r_item: list, remain_list: list, c: int, epsilon):
    u = att_result[0]
    l = len(att_result)
    r_count = list(map(int, att_result[1:l]))  # 每个目标项目增加的次数，元素转为int类型
    r_dict = dict(zip(r_item, r_count))

    # 假用户的集合
    # if sum(r_count) > (u * c):
    #     u = math.ceil(sum(r_count) / c)
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

    # fake_data = [[] for _ in range(u)]
    # min_count = int(min(r_count))
    # while min_count >= 0:
    #     for i in range(min_count):
    #         fake_data[i].append(r_item[r_count.index(min_count)])
    #     r_count[r_count.index(min_count)] = -1
    #     min_count = int(find_min_positive(r_count))

    # for x in fake_data:
    #     if len(x) < c:
    #         temp = random.sample(remain_list, c - len(x))
    #         for j in temp:
    #             x.append(j)
    #     if len(x) > c:
    #         temp = random.sample(x, c)
    #         fake_data[fake_data.index(x)] = temp

    for x in fake_data:
        if len(x) == 0:
            temp = random.sample(remain_list, 1)
            for j in temp:
                x.append(j)
    #
    # fake_data_sample = []
    # for i in range(u):
    #     temp = random.sample(fake_data[i], 1)
    #     fake_data_sample.append(temp[0])

    # g = math.ceil(math.exp(epsilon) + 1)
    #
    # # 抽样
    # fake_data_sample = [0 for col in range(u)]
    # for i in range(u):
    #     fake_data_sample[i] = fake_data[i][np.random.randint(c)]
    # # print(fake_data_sample)
    #
    # 映射
    # fake_data_hash = [0 for col in range(u)]
    # for i in range(u):
    #     fake_data_hash[i] = xxhash.xxh32(str(fake_data_sample[i]), seed=i).intdigest() % g
    # print(fake_data_hash)

    return fake_data


def data_estimate(n, epsilon, domain, Ct, fake_data, c):
    d = len(domain)
    Estimate_Dist = [0 for col in range(d)]

    g = math.ceil(math.exp(epsilon) + 1)
    p = 1 / 2
    q = 1 / g

    u = len(fake_data)
    all_n = n + u
    # print("总用户人数：", all_n)

    temp = [0 for col in range(d)]
    for i in range(d):
        for x in fake_data:
            if x == domain[i]:
                temp[i] = temp[i] + 1
    # print("temp:", temp)
    # print(sum(temp))

    # for i in range(d):
    #     t_count = 0
    #     for j in range(u):
    #         temp = xxhash.xxh32(str(domain[i]), seed=j).intdigest() % g
    #         if temp == fake_data[j]:
    #             t_count += 1
    #     Ct[i] = Ct[i] + t_count

    for i in range(d):
        Estimate_Dist[i] = c * ((Ct[i] + temp[i]) / all_n - q) / (p - q)
    return Estimate_Dist


def calculate_column_variance(matrix):
    arr = np.array(matrix)
    variances = np.var(arr, axis=0)
    return variances.tolist()

def Adjust_value(epsilon, r_item, r_true_fre, c, a):
    g = math.ceil(math.exp(epsilon) + 1)
    p = 1 / 2
    q = 1 / g

    r = len(r_item)
    D = [0 for col in range(r)]

    for i in range(r):
        D[i] = c**2 * (r_true_fre[i]*p*(1-p) + (1-r_true_fre[i])*q*(1-q)) / (n * (p-q)**2)

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



