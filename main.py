import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import asyncio


# 总体计划：
# 1.实现csv采集数据
# 2.计算指数/股票收益率
# 3.复现论文中的回归计算等
# 4.将结果整理到result.csv中
# 5.TODO：
#   1.可视化 DONE
#   2.多线程

# 时间点，用来获得有效数据
time_points = [931, 1130, 1301, 1500]

file_names = [
    "000001.csv",
    "000002.csv",
    "000004.csv",
    "000005.csv",
    "000008.csv",
    "000009.csv",
    "000010.csv",
    "000011.csv",
    "000012.csv",
    "000014.csv",
]


def cal_index_return():
    # 计算指数的收益率
    # Load data
    # 读入数据
    data = pd.read_csv("CSI_500.csv")

    final_record = []
    # time_points = [931, 1130, 1301, 1500]
    for i in range(len(data)):
        if data["time"][i] in time_points:
            final_record.append(i)
    # 计算早上的收益率
    # calculate the moring return
    return_morning = []
    for i in range(0, len(final_record), 4):
        idx = final_record[i]
        nxt_idx = final_record[i + 1]
        p1 = data["open"][idx]
        p2 = data["price"][nxt_idx]
        return_morning.append({"date": data["date"][idx], "return": (p2 - p1) / p1})
    # 计算下午的收益率
    # calculate the afternoon return
    return_afternoon = []
    for i in range(2, len(final_record), 4):
        idx = final_record[i]
        nxt_idx = final_record[i + 1]
        p1 = data["open"][idx]
        p2 = data["price"][nxt_idx]
        return_afternoon.append({"date": data["date"][idx], "return": (p2 - p1) / p1})

    return [return_morning, return_afternoon]


def cal_stock_return(file_name):
    # 计算各个股票的收益率
    # Load data
    # 读入数据
    data = pd.read_csv(file_name)

    # find the first and last records in the everyday moring
    final_record = []
    for i in range(len(data)):
        if data["time"][i] in time_points:
            final_record.append(i)
    # 计算早上的收益率
    # calculate the moring return
    return_morning = []
    for i in range(0, len(final_record), 4):
        idx = final_record[i]
        nxt_idx = final_record[i + 1]
        p1 = data["open"][idx]
        # 原本用925作为开盘价，但是有的股票在925的时候开盘价 = 0 ， 现在改成用931的开盘价
        # if p1 == 0:
        #     print("open price is 0", file_name, data["date"][idx])
        # check if the date is match
        # if data["date"][idx] != data["date"][nxt_idx]:
        #     print("date not match", file_name, data["date"][idx], data["date"][nxt_idx])
        p2 = data["price"][nxt_idx]
        return_morning.append({"date": data["date"][idx], "return": (p2 - p1) / p1})

    # 计算下午的收益率
    # calculate the afternoon return
    return_afternoon = []
    for i in range(2, len(final_record), 4):
        idx = final_record[i]
        nxt_idx = final_record[i + 1]
        p1 = data["open"][idx]
        p2 = data["price"][nxt_idx]
        # check if the date is match
        # if data["date"][idx] != data["date"][nxt_idx]:
        #     print("date not match", file_name, data["date"][idx], data["date"][nxt_idx])
        return_afternoon.append({"date": data["date"][idx], "return": (p2 - p1) / p1})

    # 计算过去20天的收益率
    ret20 = []
    for i in range(79, len(final_record), 4):
        idx = final_record[i - 80]
        nxt_idx = final_record[i]
        p1 = data["open"][idx]
        p2 = data["price"][nxt_idx]
        ret20.append({"date": data["date"][idx], "return": (p2 - p1) / p1})

    return return_morning, return_afternoon, ret20


# check the day that the index return is not in the stock return
# 测试是否有的天股票停牌，计算哪些天数，核对用
# def check_exception(index_returns, stocks_returns, name):
#     st = []
#     for i in stocks_returns:
#         st.append(i["date"])
#     for i in index_returns:
#         if i["date"] not in st:
#             print("exception", name, " ", i["date"])


def cal_epsilons(
    stock_return_morning,
    index_return_morning,
    stock_return_afternoon,
    index_return_afternoon,
):
    # 计算epsilons
    stock_returns = stock_return_morning + stock_return_afternoon
    index_returns = index_return_morning + index_return_afternoon
    data = {"stock": stock_returns, "index": index_returns}
    df = pd.DataFrame(data)
    X = df["index"]
    y = df["stock"]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    alpha = model.params[0]
    beta = model.params[1]
    epsilons_morning = []
    epsilons_afternoon = []

    # 分开计算morning和afternoon的epsilons
    for i in range(len(stock_return_morning)):
        epsilons_morning.append(
            stock_return_morning[i] - alpha - beta * index_return_morning[i]
        )
    for i in range(len(stock_return_afternoon)):
        epsilons_afternoon.append(
            stock_return_afternoon[i] - alpha - beta * index_return_afternoon[i]
        )
    # 计算epsilons的差值
    epsilons_diff = []
    for i in range(len(epsilons_morning)):
        epsilons_diff.append(epsilons_morning[i] - epsilons_afternoon[i])
    # 返回epsilons的差值
    # print(epsilons_diff)
    return epsilons_diff


def make_stat(epsilons):
    # 计算stat
    mean = np.mean(epsilons)
    std = np.std(epsilons)
    return mean / (std / np.sqrt(len(epsilons)))


def linear_analysis(stock, index_returns):
    # 线性回归分析
    final_ans = []
    # solve moring return
    index_morning = index_returns[0]
    index_afternoon = index_returns[1]
    stock_return_morning = []
    stock_return_afternoon = []
    index_return_morning = []
    index_return_afternoon = []

    # 这边使用了滑动窗口 ， 窗口长度为20，要小心停牌的情况
    j = 0
    i = 0
    while len(stock_return_morning) < 20:
        if index_morning[j]["date"] == stock["morning"][i]["date"]:
            stock_return_morning.append(stock["morning"][j]["return"])
            index_return_morning.append(index_morning[j]["return"])
            stock_return_afternoon.append(stock["afternoon"][j]["return"])
            index_return_afternoon.append(index_afternoon[j]["return"])
            i += 1
        j += 1

    ep = cal_epsilons(
        stock_return_morning,
        index_return_morning,
        stock_return_afternoon,
        index_return_afternoon,
    )
    stat = make_stat(ep)
    final_ans.append(
        {"name": stock["name"], "time": stock["morning"][i - 1]["date"], "stat": stat}
    )
    while i < len(stock["morning"]):
        if index_morning[j]["date"] != stock["morning"][i]["date"]:
            j += 1
            continue
        # 滑动窗口
        stock_return_morning.pop(0)
        index_return_morning.pop(0)
        stock_return_afternoon.pop(0)
        index_return_afternoon.pop(0)

        stock_return_morning.append(stock["morning"][i]["return"])
        index_return_morning.append(index_morning[j]["return"])
        stock_return_afternoon.append(stock["afternoon"][i]["return"])
        index_return_afternoon.append(index_afternoon[j]["return"])

        # 计算epsilons
        ep = cal_epsilons(
            stock_return_morning,
            index_return_morning,
            stock_return_afternoon,
            index_return_afternoon,
        )
        stat = make_stat(ep)
        final_ans.append(
            {"name": stock["name"], "time": stock["morning"][i]["date"], "stat": stat}
        )
        i += 1
        j += 1

    return final_ans


def cross_sectional_regression(stat, ret20):
    # 横截面回归分析
    # print(len(stat), len(ret20))

    # 处理stat 和 ret20的数据
    data_stat = []
    for i in stat:
        data_stat.append(i["stat"])
    data_ret20 = []
    for i in ret20:
        data_ret20.append(i["return"])

    data = {"stat": data_stat, "ret20": data_ret20}
    df = pd.DataFrame(data)
    X = df["stat"]
    y = df["ret20"]
    model = sm.OLS(y, X).fit()
    beta = model.params[0]
    apm_raws = []
    for i in range(len(stat)):
        # 计算APM因子
        res = ret20[i]["return"] - beta * stat[i]["stat"]
        # 这里把time 改成了date
        apm_raws.append({"name": stat[i]["name"], "date": stat[i]["time"], "res": res})
    return apm_raws


def data_to_csv(index_info, stocks_info, filename="result.csv"):
    # 保存数据到csv

    basic_count = 0  # 行戳
    columns = ["date"]
    for info in file_names:
        columns.append(info.split(".")[0])
    # 前19天因为样本不足而忽略
    stock_idx = [0 for i in range(len(file_names))]
    print(stock_idx)
    with open(filename, "w") as f:
        # 处理列名的部分
        f.write(",date")
        for info in file_names:
            f.write("," + info.split(".")[0])
        f.write("\n")
        # 处理每行的数据
        for i in range(19, len(index_info)):
            f.write((str(basic_count) + ","))
            basic_count += 1
            index_date = index_info[i]["date"]
            f.write(index_date + "")
            for j in range(len(file_names)):
                stock_date = stocks_info[j][stock_idx[j]]["date"]
                # if basic_count < 20:
                #     print(stock_date, index_date)
                if stock_date == index_date:
                    f.write("," + str(stocks_info[j][stock_idx[j]]["res"]))
                    stock_idx[j] += 1
                else:
                    # 这里是当天停牌的情况
                    # 原本准备写NONE
                    # f.write(",NONE")
                    # 最终采取的策略是用前一天的数据填充
                    f.write("," + str(stocks_info[j][stock_idx[j] - 1]["res"]))
            f.write("\n")


def visiualize(data):
    # 可视化APM因子
    for i in range(len(data)):
        x = []
        y = []
        for j in range(len(data[i])):
            # print(data[i][j]["date"], " and ", data[i][j]["res"])
            x.append(data[i][j]["date"])
            y.append(data[i][j]["res"])
        plt.plot(x, y, label=data[i][0]["name"])

    # 隐藏 x 轴的刻度
    plt.xticks([])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    index_returns = cal_index_return()

    stocks = []
    cnt = 0
    # 第1-2步
    for file_name in file_names:
        stock_return_morning, stock_return_afternoon, ret20 = cal_stock_return(
            file_name
        )
        stock_name = file_name.split(".")[0]
        # check_exception(index_return_morning, stock_return_morning, stock_name)

        # 注意这里的ret20是过去20天的综合收益率
        stocks.append(
            {
                "name": stock_name,
                "morning": stock_return_morning,
                "afternoon": stock_return_afternoon,
                "ret20": ret20,
            }
        )

        # 测试用，观察是否对应
        # code below for test
        # if len(stock_return_morning) != len(stock_return_afternoon):
        #     print("The number of stock return and index return is not equal",
        #           stock_name , len(stock_return_morning), len(stock_return_afternoon))
        # print(file_name)
        # print("Moring return")
        # for i in range(0, 10):
        #     print(stock_return_morning[i])
        # print("Afternoon return")
        # for i in range(0, 10):
        #     print(stock_return_afternoon[i])

    # 第3-5步
    stocks_infos = []
    for stock in stocks:
        # 第3-4步
        info = linear_analysis(stock, index_returns)
        ret20 = stock["ret20"]
        # 第5步
        res = cross_sectional_regression(info, ret20)
        # print(res)
        stocks_infos.append(res)
        print(stock["name"] + " Info Done")
        # print(res[0])

    # 输出到csv
    data_to_csv(index_returns[0], stocks_infos)
    print("Finished")

    # 可视化数据
    visiualize(stocks_infos)

    # break
