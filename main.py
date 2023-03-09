import json

import numpy as np
import dimod
from neal import SimulatedAnnealingSampler


class ShiftAnneal:
    def __init__(self):
        self.NAME = []  # m * 1 str
        self.DESIRE = []  # m * n int
        self.SHIFT_SIZE_LIMIT = []  # m * 1
        self.SUM_WORKDAY_LIMIT = []  # m * (1 or 2)

        self.MAN_SIZE = 0  # int
        self.DAY_SIZE = 0  # int
        self.DESIRE_LEVEL = 0  # int

        self.DESIRE_PENALTY = []  # desire levelの数 * 1 int/float
        self.SEQ_PENALTY = 0  # 連勤ルールごとに決まる
        self.SHIFT_SIZE_PENALTY = []  # d * 1
        self.SUM_WORKDAY_PENALTY = []  # m * 1

        self.NUM_READS = 0  # int

        self.liner = {}
        self.quadratic = {}
        self.sample_set = None
        self.order = None

    def getID(self, m, d):
        return self.DAY_SIZE * m + d + 100000000

    def setLIST(self, data_list: list):
        for row in data_list:
            self.NAME.append(row[0])
            self.SUM_WORKDAY_LIMIT.append(int(row[-1]))
            self.DESIRE.append([int(i) for i in row[1: -1]])
        self.MAN_SIZE = len(self.DESIRE)
        self.DAY_SIZE = len(self.DESIRE[0])

    def setParam(self, des_const, seq_const, shift_size_const, shift_size_limit: list, workday_const, workday: list,
                 num_reads):
        self.DESIRE_PENALTY = des_const
        self.SEQ_PENALTY = seq_const
        self.SHIFT_SIZE_PENALTY = shift_size_const
        self.SHIFT_SIZE_LIMIT = shift_size_limit
        self.SUM_WORKDAY_PENALTY = workday_const
        # self.WORKDAY = workday
        self.NUM_READS = num_reads

    def setConst(self):
        # １次
        for i in range(self.MAN_SIZE):
            for j in range(self.DAY_SIZE):
                liner_const = (self.DESIRE[i][j] * self.DESIRE_PENALTY)  # 出勤希望度による
                liner_const -= 2 * self.SHIFT_SIZE_LIMIT[j] * self.SHIFT_SIZE_PENALTY  # １シフトに入る人数制約による
                liner_const -= 2 * self.SUM_WORKDAY_LIMIT[i] * self.SUM_WORKDAY_PENALTY  # 勤務日数希望による
                key = "x_{0}".format(self.getID(i, j))
                try:
                    self.liner[key] += liner_const
                except KeyError:
                    self.liner[key] = liner_const

        # ２次
        # 昼夜連勤の禁止による
        for i in range(self.MAN_SIZE):
            for j in range(int(self.DAY_SIZE / 2)):
                j *= 2
                key = ("x_{0}".format(self.getID(i, j)), "x_{0}".format(self.getID(i, j + 1)))
                try:
                    self.quadratic[key] += self.SEQ_PENALTY
                except KeyError:
                    self.quadratic[key] = self.SEQ_PENALTY

        # １シフトに入る人数制約による
        for i1 in range(self.MAN_SIZE):
            for i2 in range(i1, self.MAN_SIZE):
                if i1 == i2:
                    for j in range(self.DAY_SIZE):
                        key = ("x_{0}".format(self.getID(i1, j)), "x_{0}".format(self.getID(i2, j)))
                        try:
                            self.quadratic[key] += 1 * self.SHIFT_SIZE_PENALTY
                        except KeyError:
                            self.quadratic[key] = 1 * self.SHIFT_SIZE_PENALTY
                else:
                    for j in range(self.DAY_SIZE):
                        key = ("x_{0}".format(self.getID(i1, j)), "x_{0}".format(self.getID(i2, j)))
                        try:
                            self.quadratic[key] += 2 * self.SHIFT_SIZE_PENALTY
                        except KeyError:
                            self.quadratic[key] = 2 * self.SHIFT_SIZE_PENALTY

        # 勤務日数希望による
        for j1 in range(self.DAY_SIZE):
            for j2 in range(j1, self.DAY_SIZE):
                if j1 == j2:
                    for i in range(self.MAN_SIZE):
                        key = ("x_{0}".format(self.getID(i, j1)), "x_{0}".format(self.getID(i, j2)))
                        try:
                            self.quadratic[key] += 1 * self.SUM_WORKDAY_PENALTY
                        except KeyError:
                            self.quadratic[key] = 1 * self.SUM_WORKDAY_PENALTY
                else:
                    for i in range(self.MAN_SIZE):
                        key = ("x_{0}".format(self.getID(i, j1)), "x_{0}".format(self.getID(i, j2)))
                        try:
                            self.quadratic[key] += 2 * self.SUM_WORKDAY_PENALTY
                        except KeyError:
                            self.quadratic[key] = 2 * self.SUM_WORKDAY_PENALTY

    def setName(self, name_list: list):
        self.NAME = []
        for name in name_list:
            if name in self.NAME:
                print("Error: 同じ名前の人が複数存在します。")
            elif (not name) or (len(name) > 10):
                print("Error: 名前は1文字以上10文字以下で登録してください。")
            else:
                self.NAME.append(str(name))

        self.MAN_SIZE = len(self.NAME)

    def setDesire(self, desire_list: list):
        self.DESIRE = []
        if not self.MAN_SIZE:
            print("Error: 名前の登録を行ってから、希望度の設定をしてください。")
        elif len(desire_list) != self.MAN_SIZE:
            print("Error: 希望度の行数が設定された名前の数と一致しません。")
        else:
            for desire in desire_list:
                if type(desire) != list:
                    print("Error: 希望度は２次元配列である必要があります。")
                elif not self.DAY_SIZE:
                    self.DAY_SIZE = len(desire)  # 最初だけ
                elif len(desire) != self.DAY_SIZE:
                    print("Error: 希望度の列数が統一されていません。")
                else:
                    for des in desire:
                        if type(des) != int or (type(des) == int and des < 0):
                            print("Error: 希望度は非負整数である必要があります。")
                        else:
                            self.DESIRE.append(desire)

    def setShift_Size_Limit(self, ssl_list: list):
        self.SHIFT_SIZE_LIMIT = []
        if not self.DAY_SIZE:
            print("Error: 希望度の登録を行ってから、シフトサイズの設定をしてください。")
        elif len(ssl_list) != self.DAY_SIZE:
            print("Error: シフトサイズの数と希望度の列数が一致しません。")
        else:
            for ssl in ssl_list:
                if type(ssl) != int or (type(ssl) == int and ssl < 0):
                    print("Error: シフトサイズは非負整数である必要があります。")
                else:
                    self.SHIFT_SIZE_LIMIT.append(ssl)

    def setSum_Workday_Limit(self, sw_list: list or int):
        if not self.MAN_SIZE:
            print("Error: 名前の登録を行ってから、勤務日数希望の設定をしてください。")
        elif len(sw_list) != self.MAN_SIZE:
            print("Error: 勤務日数希望の数と名前の数が一致しません。")
        else:
            for sw in sw_list:
                if (type(sw) != list and type(sw) != int) or (
                        type(sw) == list and len(sw) != 2) or (
                        type(sw) == int and (not 0 <= sw <= self.DAY_SIZE)
                ):
                    print("Error: 勤務日数希望は0以上希望度の列数以下の非負整数もしくは配列（要素数2）である必要があります。")
                elif type(sw) == list and (type(sw[0]) != int or type(sw[1]) != int):
                    print("Error: 勤務日数希望は0以上希望度の列数以下の非負整数もしくは配列（要素数2）である必要があります。")
                elif type(sw) == list and (not 0 <= sw[0] <= self.DAY_SIZE or not 0 <= sw[1] <= self.DAY_SIZE):
                    print("Error: 勤務日数希望は0以上希望度の列数以下の非負整数もしくは配列（要素数2）である必要があります。")
                else:
                    self.SUM_WORKDAY_LIMIT.append(sw)

    def setDesire_Penalty(self, desire_penalty_list: list):
        if not self.DESIRE:
            print("Error: 希望度の設定を行ってから、希望度のペナルティを設定してください。")
        elif len(desire_penalty_list) - 1 < self.DESIRE_LEVEL:
            print("Error: 設定された希望度に対し、ペナルティの数が足りません。")
        else:
            for desire_pena in desire_penalty_list:
                if type(desire_pena) != int or (type(desire_pena) == int and desire_pena < 0):
                    print("Error: 希望度のペナルティ値は非負整数である必要があります。")
                else:
                    self.DESIRE_PENALTY.append(desire_pena)

    def setSeq_Penalty(self, seq_penalty_list):
        if not self.MAN_SIZE:
            print("Error: 名前の登録を行ってから、連勤のペナルティ値を設定してください。")
        else:
            for seq_penalty in seq_penalty_list:
                if type(seq_penalty) != list or (type(seq_penalty) == list and len(seq_penalty) != 3):
                    print("Error: 連勤のペナルティは「対象者(配列)」「制限するパターン(配列)」「ペナルティ係数」の３つが必要です。")
                elif type(seq_penalty[0]) != list:
                    print("Error: 連勤のペナルティの「対象者」は配列である必要があります。")
                elif type(seq_penalty[1]) != list or (type(seq_penalty[1]) == list and len(seq_penalty[1]) != 2):
                    print("Error: 連勤のペナルティの「制限するパターン」は要素数2の配列である必要があります。")
                elif type(seq_penalty[2]) != int or (type(seq_penalty[2]) == int and seq_penalty[2] < 0):
                    print("Error: 連勤のペナルティの「ペナルティ係数」は非負整数である必要があります。")
                else:
                    self.SEQ_PENALTY.append(seq_penalty)

    def setShift_Size_Penalty(self, ssl_penalty_list: list):
        if not self.DAY_SIZE:
            print("Error: 希望度の設定を行ってから、シフトサイズのペナルティを設定してください。")
        elif len(ssl_penalty_list) != self.DAY_SIZE:
            print("Error: シフトサイズのペナルティの数と希望度の列数が一致しません。")
        else:
            for ssl_pena in ssl_penalty_list:
                if type(ssl_pena) != int or (type(ssl_pena) == int and ssl_pena < 0):
                    print("Error: シフトサイズのペナルティ値は非負整数である必要があります。")
                else:
                    self.SHIFT_SIZE_PENALTY.append(ssl_pena)

    def setSum_Workday_Penalty(self, sw_penalty_list: list):
        if not self.NAME:
            print("Error: 名前の設定を行ってから、勤務日数希望のペナルティを設定してください。")
        elif len(sw_penalty_list) != self.MAN_SIZE:
            print("Error: 勤務日数希望のペナルティの数と名前の数が一致しません。")
        else:
            for sw_pena in sw_penalty_list:
                if type(sw_pena) != int or (type(sw_pena) == int and sw_pena < 0):
                    print("Error: 勤務日数希望のペナルティ値は非負整数である必要があります。")
                else:
                    self.SUM_WORKDAY_PENALTY.append(sw_pena)

    def addDesire_Constraint(self):
        if not all([self.MAN_SIZE, self.DAY_SIZE, self.DESIRE, self.DESIRE_PENALTY]):
            print("Error: データとパラメータの設定を行ってから、希望度の制約式を設定してください。")
        else:
            for i in range(self.MAN_SIZE):
                for j in range(self.DAY_SIZE):
                    key = "x_{0}".format(self.getID(i, j))
                    liner_const = self.DESIRE_PENALTY[self.DESIRE[i][j]]
                    try:
                        self.liner[key] += liner_const
                    except KeyError:
                        self.liner[key] = liner_const

    def addSeq_Constraint(self):
        if not all([self.MAN_SIZE, self.DAY_SIZE]):
            print("Error: データとパラメータの設定を行ってから、連勤の制約式を設定してください。")

    def addShift_Size_Constraint(self):
        if not all([self.MAN_SIZE, self.DAY_SIZE, self.SHIFT_SIZE_LIMIT, self.SHIFT_SIZE_PENALTY]):
            print("Error: データとパラメータの設定を行ってから、シフトサイズの制約式を設定してください。")

        for d in range(self.DAY_SIZE):
            # １次
            for m in range(self.MAN_SIZE):
                key = "x_{0}".format(self.getID(m, d))
                liner_const = - 2 * self.SHIFT_SIZE_LIMIT[d] * self.SHIFT_SIZE_PENALTY[d]  # １シフトに入る人数制約による
                try:
                    self.liner[key] += liner_const
                except KeyError:
                    self.liner[key] = liner_const
            # ２次
            for m1 in range(self.MAN_SIZE):
                for m2 in range(m1, self.MAN_SIZE):
                    key = ("x_{0}".format(self.getID(m1, d)), "x_{0}".format(self.getID(m2, d)))
                    if m1 == m2:
                        quad_const = 1 * self.SHIFT_SIZE_PENALTY[d]
                    else:
                        quad_const = 2 * self.SHIFT_SIZE_PENALTY[d]
                    try:
                        self.quadratic[key] += quad_const
                    except KeyError:
                        self.quadratic[key] = quad_const

    def addSum_Workday_Constraint(self):
        if not all([self.MAN_SIZE, self.DAY_SIZE, self.SUM_WORKDAY_LIMIT, self.SUM_WORKDAY_PENALTY]):
            print("Error: データとパラメータの設定を行ってから、シフトサイズの制約式を設定してください。")

        for m in range(self.MAN_SIZE):
            limit = self.SUM_WORKDAY_LIMIT
            if type(limit) == int:
                limit = [limit, self.DAY_SIZE]

            if limit[1] == 0:  # 分母がゼロの時は、勤務日数希望なしとして扱う
                return

            num_unit = self.DAY_SIZE - limit[1] + 1

            for day_unit_id in range(0, num_unit):
                # 1次
                for d in range(day_unit_id, day_unit_id + limit[1]):
                    key = "x_{0}".format(self.getID(m, d))
                    liner_const = - 2 / num_unit * self.SUM_WORKDAY_LIMIT[m] * self.SUM_WORKDAY_PENALTY[m]
                    try:
                        self.liner[key] += liner_const
                    except KeyError:
                        self.liner[key] = liner_const
                # 2次
                for d1 in range(day_unit_id, day_unit_id + limit[1]):
                    for d2 in range(d1, day_unit_id + limit[1]):
                        key = ("x_{0}".format(self.getID(m, d1)), "x_{0}".format(self.getID(m, d2)))
                        if d1 == d2:
                            quad_const = 1 / num_unit * self.SUM_WORKDAY_PENALTY[m]
                        else:
                            quad_const = 2 / num_unit * self.SUM_WORKDAY_PENALTY[m]
                        try:
                            self.quadratic[key] += quad_const
                        except KeyError:
                            self.quadratic[key] = quad_const

    def setConstraint(self):
        pass

    def sample(self):
        bqm = dimod.BinaryQuadraticModel(self.liner, self.quadratic, 0, "BINARY")
        SA_sampler = SimulatedAnnealingSampler()
        self.sample_set = SA_sampler.sample(bqm, num_reads=self.NUM_READS, beta_schedule_type="geometric",
                                            num_sweeps_per_beta=100, num_sweeps=10000)
        self.order = np.argsort(self.sample_set.record["energy"])

    def getResult(self):
        ret = []
        first = self.sample_set.record[self.order][0][0]
        for man in range(self.MAN_SIZE):
            tmp = [self.NAME[man]]
            for day in range(self.DAY_SIZE):
                tmp.append(int(first[man * self.DAY_SIZE + day]))  # ここのint()はjson化するときのために必要
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
