import json

import numpy as np
import dimod
from neal import SimulatedAnnealingSampler


class ShiftAnneal:
    def __init__(self):
        self.NAME = []
        self.MANPOWER = 0
        self.DAY = 0
        self.DESIRE_CONST = 0
        self.SEQ_CONST = 0
        self.SHIFT_SIZE_CONST = 0
        self.SHIFT_SIZE_LIMIT = []
        self.WORKDAY = []
        self.WORKDAY_CONST = 0
        self.NUM_READS = 0

        self.const = []
        self.liner = {}
        self.quadratic = {}
        self.sample_set = None
        self.order = None

    def getID(self, m, d):
        return self.DAY * m + d + 100000000

    def setLIST(self, data_list: list):
        for row in data_list:
            self.NAME.append(row[0])
            self.WORKDAY.append(int(row[-1]))
            self.const.append([int(i) for i in row[1: -1]])
        self.MANPOWER = len(self.const)
        self.DAY = len(self.const[0])

    def setParam(self, des_const, seq_const, shift_size_const, shift_size_limit: list, workday_const, workday: list,
                 num_reads):
        self.DESIRE_CONST = des_const
        self.SEQ_CONST = seq_const
        self.SHIFT_SIZE_CONST = shift_size_const
        self.SHIFT_SIZE_LIMIT = shift_size_limit
        self.WORKDAY_CONST = workday_const
        # self.WORKDAY = workday
        self.NUM_READS = num_reads

    def setConst(self):
        # １次
        for i in range(self.MANPOWER):
            for j in range(self.DAY):
                liner_const = (self.const[i][j] * self.DESIRE_CONST)  # 出勤希望度による
                liner_const -= 2 * self.SHIFT_SIZE_LIMIT[j] * self.SHIFT_SIZE_CONST  # １シフトに入る人数制約による
                liner_const -= 2 * self.WORKDAY[i] * self.WORKDAY_CONST  # 勤務日数希望による
                key = "x_{0}".format(self.getID(i, j))
                try:
                    self.liner[key] += liner_const
                except KeyError:
                    self.liner[key] = liner_const

        # ２次
        # 昼夜連勤の禁止による
        for i in range(self.MANPOWER):
            for j in range(int(self.DAY / 2)):
                j *= 2
                key = ("x_{0}".format(self.getID(i, j)), "x_{0}".format(self.getID(i, j + 1)))
                try:
                    self.quadratic[key] += self.SEQ_CONST
                except KeyError:
                    self.quadratic[key] = self.SEQ_CONST

        # １シフトに入る人数制約による
        for i1 in range(self.MANPOWER):
            for i2 in range(i1, self.MANPOWER):
                if i1 == i2:
                    for j in range(self.DAY):
                        key = ("x_{0}".format(self.getID(i1, j)), "x_{0}".format(self.getID(i2, j)))
                        try:
                            self.quadratic[key] += 1 * self.SHIFT_SIZE_CONST
                        except KeyError:
                            self.quadratic[key] = 1 * self.SHIFT_SIZE_CONST
                else:
                    for j in range(self.DAY):
                        key = ("x_{0}".format(self.getID(i1, j)), "x_{0}".format(self.getID(i2, j)))
                        try:
                            self.quadratic[key] += 2 * self.SHIFT_SIZE_CONST
                        except KeyError:
                            self.quadratic[key] = 2 * self.SHIFT_SIZE_CONST

        # 勤務日数希望による
        for j1 in range(self.DAY):
            for j2 in range(j1, self.DAY):
                if j1 == j2:
                    for i in range(self.MANPOWER):
                        key = ("x_{0}".format(self.getID(i, j1)), "x_{0}".format(self.getID(i, j2)))
                        try:
                            self.quadratic[key] += 1 * self.WORKDAY_CONST
                        except KeyError:
                            self.quadratic[key] = 1 * self.WORKDAY_CONST
                else:
                    for i in range(self.MANPOWER):
                        key = ("x_{0}".format(self.getID(i, j1)), "x_{0}".format(self.getID(i, j2)))
                        try:
                            self.quadratic[key] += 2 * self.WORKDAY_CONST
                        except KeyError:
                            self.quadratic[key] = 2 * self.WORKDAY_CONST

    def sample(self):
        # BQMモデルに変換
        bqm = dimod.BinaryQuadraticModel(self.liner, self.quadratic, 0, "BINARY")
        # サンプリング
        SA_sampler = SimulatedAnnealingSampler()
        self.sample_set = SA_sampler.sample(bqm, num_reads=self.NUM_READS, beta_schedule_type="geometric",
                                            num_sweeps_per_beta=100, num_sweeps=10000)
        self.order = np.argsort(self.sample_set.record["energy"])

    def getResult(self):
        ret = []
        first = self.sample_set.record[self.order][0][0]
        for man in range(self.MANPOWER):
            tmp = [self.NAME[man]]
            for day in range(self.DAY):
                tmp.append(int(first[man * self.DAY + day]))  # ここのint()はjson化するときのために必要
            ret.append(tmp)
        return ret


def main(request):
    request_json = request.get_json()
    if request_json and "list" in request_json and "param" in request_json:
        first = optimize(request_json["list"], request_json["param"])
        return json.dumps({"State": "success", "result": first})
    else:
        return json.dumps({"State": "error"})


def optimize(ls, pr):
    model = ShiftAnneal()
    model.setLIST(data_list=ls)
    model.setParam(des_const=6 * pr[1][0],
                   seq_const=30 * pr[1][1],
                   shift_size_const=30 * pr[1][2],
                   workday_const=1 * pr[1][3],
                   shift_size_limit=pr[0],
                   workday=["ignore"],  # setLISTで設定しているから使わない。
                   num_reads=100)
    model.setConst()
    model.sample()
    return model.getResult()
