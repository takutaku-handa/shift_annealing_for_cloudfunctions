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
        self.SEQ_PENALTY = []  # 連勤ルールごとに決まる ? * 3 * ?
        self.SHIFT_SIZE_PENALTY = []  # d * 1
        self.SUM_WORKDAY_PENALTY = []  # m * 1

        self.NUM_READS = 10  # int

        self.liner = {}
        self.quadratic = {}
        self.sample_set = None
        self.order = None

        self.message = ""

    def getID(self, m, d):
        return self.DAY_SIZE * m + d + 100000000

    def setMessage(self, mes):
        self.message = mes

    def setNumReads(self, num):
        if type(num) != int or (type(num) == int and num < 1):
            self.setMessage("Error: num_readsは正の整数である必要があります。")
        elif num > 5000:
            self.NUM_READS = 5000
        else:
            self.NUM_READS = num

    def setName(self, name_list: list):
        if self.message:
            return
        self.NAME = []
        for name in name_list:
            if name in self.NAME:
                self.setMessage("Error: 同じ名前の人が複数存在します。")
            elif (not name) or (len(name) > 10):
                self.setMessage("Error: 名前は1文字以上10文字以下で登録してください。")
            else:
                self.NAME.append(str(name))

        self.MAN_SIZE = len(self.NAME)

    def setDesire(self, desire_list: list):
        if self.message:
            return
        self.DESIRE = []
        if not self.MAN_SIZE:
            self.setMessage("Error: 名前の登録を行ってから、希望度の設定をしてください。")
        elif len(desire_list) != self.MAN_SIZE:
            self.setMessage("Error: 希望度の行数が設定された名前の数と一致しません。")
        else:
            for desire in desire_list:
                if type(desire) != list:
                    self.setMessage("Error: 希望度は２次元配列である必要があります。")
                    return
                elif not self.DAY_SIZE:
                    self.DAY_SIZE = len(desire)  # 最初だけ
                elif len(desire) != self.DAY_SIZE:
                    self.setMessage("Error: 希望度の列数が統一されていません。")
                    return
                else:
                    for des in desire:
                        if type(des) != int or (type(des) == int and des < 0):
                            self.setMessage("Error: 希望度は非負整数である必要があります。")
                            return
                        else:
                            self.DESIRE_LEVEL = max(self.DESIRE_LEVEL, max(desire))
                self.DESIRE.append(desire)

    def setShift_Size_Limit(self, ssl_list: list):
        if self.message:
            return
        self.SHIFT_SIZE_LIMIT = []
        if not self.DAY_SIZE:
            self.setMessage("Error: 希望度の登録を行ってから、シフトサイズの設定をしてください。")
        elif len(ssl_list) != self.DAY_SIZE:
            self.setMessage("Error: シフトサイズの数と希望度の列数が一致しません。")
        else:
            for ssl in ssl_list:
                if type(ssl) != int or (type(ssl) == int and ssl < 0):
                    self.setMessage("Error: シフトサイズは非負整数である必要があります。")
                else:
                    self.SHIFT_SIZE_LIMIT.append(ssl)

    def setSum_Workday_Limit(self, swl_list: list or int):
        if self.message:
            return
        if not self.MAN_SIZE:
            self.setMessage("Error: 名前の登録を行ってから、勤務日数希望の設定をしてください。")
        elif len(swl_list) != self.MAN_SIZE:
            self.setMessage("Error: 勤務日数希望の数と名前の数が一致しません。")
        else:
            for sw in swl_list:
                if (type(sw) != list and type(sw) != int) or (
                        type(sw) == list and len(sw) != 2) or (
                        type(sw) == int and (not 0 <= sw <= self.DAY_SIZE)
                ):
                    self.setMessage("Error: 勤務日数希望は0以上希望度の列数以下の非負整数もしくは配列（要素数2）である必要があります。")
                elif type(sw) == list and (type(sw[0]) != int or type(sw[1]) != int):
                    self.setMessage("Error: 勤務日数希望は0以上希望度の列数以下の非負整数もしくは配列（要素数2）である必要があります。")
                elif type(sw) == list and (not 0 <= sw[0] <= self.DAY_SIZE or not 0 <= sw[1] <= self.DAY_SIZE):
                    self.setMessage("Error: 勤務日数希望は0以上希望度の列数以下の非負整数もしくは配列（要素数2）である必要があります。")
                else:
                    self.SUM_WORKDAY_LIMIT.append(sw)

    def setDesire_Penalty(self, desire_penalty_list: list):
        if self.message:
            return
        if not self.DESIRE:
            self.setMessage("Error: 希望度の設定を行ってから、希望度のペナルティを設定してください。")
        elif len(desire_penalty_list) < self.DESIRE_LEVEL + 1:
            self.setMessage("Error: 設定された希望度に対し、ペナルティの数が足りません。")
        else:
            for desire_pena in desire_penalty_list:
                if type(desire_pena) != int or (type(desire_pena) == int and desire_pena < 0):
                    self.setMessage("Error: 希望度のペナルティ値は非負整数である必要があります。")
                else:
                    self.DESIRE_PENALTY.append(desire_pena)

    def setSeq_Penalty(self, seq_penalty_list):
        if self.message:
            return
        if not self.MAN_SIZE:
            self.setMessage("Error: 名前の登録を行ってから、連勤のペナルティ値を設定してください。")
        else:
            for seq_penalty in seq_penalty_list:
                if type(seq_penalty) != list or (type(seq_penalty) == list and len(seq_penalty) != 3):
                    self.setMessage("Error: 連勤のペナルティは「対象者(配列)」「制限するパターン(配列)」「ペナルティ係数」の３つが必要です。")
                elif type(seq_penalty[0]) != list:
                    self.setMessage("Error: 連勤のペナルティの「対象者」は配列である必要があります。")
                elif type(seq_penalty[1]) != list or (type(seq_penalty[1]) == list and len(seq_penalty[1]) != 2):
                    self.setMessage("Error: 連勤のペナルティの「制限するパターン」は要素数2の配列である必要があります。")
                elif type(seq_penalty[2]) != int or (type(seq_penalty[2]) == int and seq_penalty[2] < 0):
                    self.setMessage("Error: 連勤のペナルティの「ペナルティ係数」は非負整数である必要があります。")
                else:
                    self.SEQ_PENALTY.append(seq_penalty)

    def setShift_Size_Penalty(self, ssl_penalty_list: list):
        if self.message:
            return
        if not self.DAY_SIZE:
            self.setMessage("Error: 希望度の設定を行ってから、シフトサイズのペナルティを設定してください。")
        elif len(ssl_penalty_list) != self.DAY_SIZE:
            self.setMessage("Error: シフトサイズのペナルティの数と希望度の列数が一致しません。")
        else:
            for ssl_pena in ssl_penalty_list:
                if type(ssl_pena) != int or (type(ssl_pena) == int and ssl_pena < 0):
                    self.setMessage("Error: シフトサイズのペナルティ値は非負整数である必要があります。")
                else:
                    self.SHIFT_SIZE_PENALTY.append(ssl_pena)

    def setSum_Workday_Penalty(self, sw_penalty_list: list):
        if self.message:
            return
        if not self.NAME:
            self.setMessage("Error: 名前の設定を行ってから、勤務日数希望のペナルティを設定してください。")
        elif len(sw_penalty_list) != self.MAN_SIZE:
            self.setMessage("Error: 勤務日数希望のペナルティの数と名前の数が一致しません。")
        else:
            for sw_pena in sw_penalty_list:
                if type(sw_pena) != int or (type(sw_pena) == int and sw_pena < 0):
                    self.setMessage("Error: 勤務日数希望のペナルティ値は非負整数である必要があります。")
                else:
                    self.SUM_WORKDAY_PENALTY.append(sw_pena)

    def setRequiredData(self, name, desire, shift_size_limit, sum_workday_limit):
        self.setName(name)
        self.setDesire(desire)
        self.setShift_Size_Limit(shift_size_limit)
        self.setSum_Workday_Limit(sum_workday_limit)

    def setLiner(self, key, const):
        try:
            self.liner[key] += const
        except KeyError:
            self.liner[key] = const

    def setQuadratic(self, key, const):
        try:
            self.quadratic[key] += const
        except KeyError:
            self.quadratic[key] = const

    def addDesire_Constraint(self):
        if self.message:
            return
        if not all([self.MAN_SIZE, self.DAY_SIZE, self.DESIRE, self.DESIRE_PENALTY]):
            self.setMessage("Error: データとパラメータの設定を行ってから、希望度の制約式を設定してください。")
        else:
            for m in range(self.MAN_SIZE):
                target_d = []
                for d in range(self.DAY_SIZE):
                    if self.DESIRE[m][d]:
                        target_d.append(d)
                for d1 in range(len(target_d)):
                    for d2 in range(d1, len(target_d)):
                        key = ("x_{0}".format(self.getID(m, d1)), "x_{0}".format(self.getID(m, d2)))
                        if d1 == d2:
                            quad_const = self.DESIRE_PENALTY[self.DESIRE[m][d1]] * \
                                         self.DESIRE_PENALTY[self.DESIRE[m][d2]]
                        else:
                            quad_const = 2 * self.DESIRE_PENALTY[self.DESIRE[m][d1]] * \
                                         self.DESIRE_PENALTY[self.DESIRE[m][d2]]
                        self.setQuadratic(key, quad_const)

    def addSeq_Constraint(self):
        if self.message:
            return
        if not all([self.MAN_SIZE, self.DAY_SIZE]):
            self.setMessage("Error: データとパラメータの設定を行ってから、連勤の制約式を設定してください。")

        for seq_pena in self.SEQ_PENALTY:
            target = seq_pena[0]
            if target == ["all"]:
                target = [_ for _ in range(self.MAN_SIZE)]
            for t in target:
                if type(t) != int or (type(t) == int and t < 0):
                    self.setMessage("Error: 連勤制約の対象者は、非負整数の配列もしくは['all']である必要があります。")
                    return

            coefficient = []
            for pat in seq_pena[1]:
                try:
                    a = int(pat.split("i")[0])
                    b = int(pat.split("i")[1])
                except ValueError or IndexError or AttributeError:
                    self.setMessage("Error: 連勤制約のパターンの記述ルール(ex: -2i+4, 1i-11)に従っていません。")
                    return
                else:
                    coefficient.append([a, b])

            for i in range(self.DAY_SIZE):
                d1 = coefficient[0][0] * i + coefficient[0][1]
                d2 = coefficient[1][0] * i + coefficient[1][1]
                if d1 < 0 or d2 < 0:
                    continue
                if self.DAY_SIZE <= d1 or self.DAY_SIZE <= d2:
                    break
                for m in target:
                    key = ("x_{0}".format(self.getID(m, d1)), "x_{0}".format(self.getID(m, d2)))
                    self.setQuadratic(key, seq_pena[2])

    def addShift_Size_Constraint(self):
        if self.message:
            return
        if not all([self.MAN_SIZE, self.DAY_SIZE, self.SHIFT_SIZE_LIMIT, self.SHIFT_SIZE_PENALTY]):
            self.setMessage("Error: データとパラメータの設定を行ってから、シフトサイズの制約式を設定してください。")
            return

        for d in range(self.DAY_SIZE):
            # １次
            for m in range(self.MAN_SIZE):
                key = "x_{0}".format(self.getID(m, d))
                liner_const = - 2 * self.SHIFT_SIZE_LIMIT[d] * self.SHIFT_SIZE_PENALTY[d]  # １シフトに入る人数制約による
                self.setLiner(key, liner_const)
            # ２次
            for m1 in range(self.MAN_SIZE):
                for m2 in range(m1, self.MAN_SIZE):
                    key = ("x_{0}".format(self.getID(m1, d)), "x_{0}".format(self.getID(m2, d)))
                    if m1 == m2:
                        quad_const = 1 * self.SHIFT_SIZE_PENALTY[d]
                    else:
                        quad_const = 2 * self.SHIFT_SIZE_PENALTY[d]
                    self.setQuadratic(key, quad_const)

    def addSum_Workday_Constraint(self):
        if self.message:
            return
        if not all([self.MAN_SIZE, self.DAY_SIZE, self.SUM_WORKDAY_LIMIT, self.SUM_WORKDAY_PENALTY]):
            self.setMessage("Error: データとパラメータの設定を行ってから、シフトサイズの制約式を設定してください。")
            return

        for m in range(self.MAN_SIZE):
            limit = self.SUM_WORKDAY_LIMIT[m]
            if type(limit) == int:
                limit = [limit, self.DAY_SIZE]

            if limit[1] == 0:  # 分母がゼロの時は、勤務日数希望なしとして扱う
                return

            num_unit = self.DAY_SIZE - limit[1] + 1

            for day_unit_id in range(0, num_unit):
                # 1次
                for d in range(day_unit_id, day_unit_id + limit[1]):
                    key = "x_{0}".format(self.getID(m, d))
                    liner_const = - 2 / num_unit * limit[0] * self.SUM_WORKDAY_PENALTY[m]
                    self.setLiner(key, liner_const)
                # 2次
                for d1 in range(day_unit_id, day_unit_id + limit[1]):
                    for d2 in range(d1, day_unit_id + limit[1]):
                        key = ("x_{0}".format(self.getID(m, d1)), "x_{0}".format(self.getID(m, d2)))
                        if d1 == d2:
                            quad_const = 1 / num_unit * self.SUM_WORKDAY_PENALTY[m]
                        else:
                            quad_const = 2 / num_unit * self.SUM_WORKDAY_PENALTY[m]
                        self.setQuadratic(key, quad_const)

    def addRequiredConstraints(self):
        self.addDesire_Constraint()
        self.addSeq_Constraint()
        self.addShift_Size_Constraint()
        self.addSum_Workday_Constraint()

    def sample(self):
        if self.message:
            return
        bqm = dimod.BinaryQuadraticModel(self.liner, self.quadratic, 0, "BINARY")
        SA_sampler = SimulatedAnnealingSampler()
        self.sample_set = SA_sampler.sample(bqm, num_reads=self.NUM_READS, beta_schedule_type="geometric",
                                            num_sweeps_per_beta=100, num_sweeps=10000)
        self.order = np.argsort(self.sample_set.record["energy"])

    def getResult(self):
        if self.message:
            return [self.message, []]
        ret = []
        first = self.sample_set.record[self.order][0][0]
        for man in range(self.MAN_SIZE):
            tmp = [self.NAME[man]]
            for day in range(self.DAY_SIZE):
                tmp.append(int(first[man * self.DAY_SIZE + day]))  # ここのint()はjson化するときのために必要
            ret.append(tmp)
        return [self.message, ret]


def optimize(req_json):
    model = ShiftAnneal()
    try:
        name = req_json["name"]
        desire = req_json["desire"]
        shift_size_limit = req_json["shift_size_limit"]
        sum_workday_limit = req_json["sum_workday_limit"]
    except KeyError:
        model.setMessage("Error: 名前, 出勤希望度, シフトサイズ, 勤務日数希望 は必須です。")
    else:
        model.setRequiredData(name, desire, shift_size_limit, sum_workday_limit)
        if "desire_penalty" in req_json.keys():
            model.setDesire_Penalty(req_json["desire_penalty"])
            model.addDesire_Constraint()
        if "shift_size_penalty" in req_json.keys():
            model.setShift_Size_Penalty(req_json["shift_size_penalty"])
            model.addShift_Size_Constraint()
        if "sum_workday_penalty" in req_json.keys():
            model.setSum_Workday_Penalty(req_json["sum_workday_penalty"])
            model.addSum_Workday_Constraint()
        if "seq_penalty" in req_json.keys():
            model.setSeq_Penalty(req_json["seq_penalty"])
            model.addSeq_Constraint()
        if "num_reads" in req_json.keys():
            model.setNumReads(req_json["num_reads"])
        model.sample()
    return model.getResult()


def main(request):
    request_json = request.get_json()
    result = optimize(request_json)
    return json.dumps({"state": result[0], "result": result[1]}, ensure_ascii=False)
