import csv
import math
import numpy as np
import dimod
from neal import SimulatedAnnealingSampler
from linebot import LineBotApi
from linebot.models import TextSendMessage


def hello_world(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """
    request_json = request.get_json()
    if request_json and 'message' in request_json:
        print(request_json['message'])
        main(request_json['message'])
        return 'Got List!'
    else:
        print("Hello World")
        return 'Hello World!'


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

    def setCSV(self, filename):
        with open(filename, encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                self.NAME.append(row[0])
                self.WORKDAY.append(int(row[-1]))
                self.const.append([int(i) for i in row[1: -1]])

        self.MANPOWER = len(self.const)
        self.DAY = len(self.const[0])

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
        self.WORKDAY = workday
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

    def getPenalty(self, sample):
        # ペナルティ：出勤希望度違反数、昼夜連勤数、人数超過日数、人数不足日数、勤務日のばらつき、勤務日数のばらつき
        pena_desire = [0, 0, 0, 0]
        pena_seq = 0
        pena_over = 0
        pena_lack = 0
        pena_dist = 0
        pena_workday = 0

        count_horizontal = [0 for _ in range(self.MANPOWER)]
        count_vertical = [0 for _ in range(self.DAY)]
        max_seq_work = 0
        max_seq_off = 0

        index = 0
        for m in range(self.MANPOWER):
            work_date = []
            date = 1
            seq_work = 0
            seq_off = 0
            for d in range(self.DAY):
                s = sample[index]
                if s:
                    work_date.append(date)
                    seq_work += 1
                    seq_off = 0
                else:
                    seq_work = 0
                    seq_off += 1
                if seq_work > max_seq_work:
                    max_seq_work = seq_work
                if seq_off > max_seq_off:
                    max_seq_off = seq_off
                date += 1
                pena_desire[self.const[m][d]] += s
                if (d % self.DAY) % 2 == 0 and d < self.DAY and s * sample[index + 1]:
                    pena_seq += 1
                count_horizontal[m] += s
                count_vertical[d] += s
                index += 1
            pena_dist += math.sqrt(np.var(work_date))

        for cv in range(self.DAY):
            if count_vertical[cv] > self.SHIFT_SIZE_LIMIT[cv]:
                pena_over += 1
            if count_vertical[cv] < self.SHIFT_SIZE_LIMIT[cv]:
                pena_lack += 1

        for m in range(self.MANPOWER):
            gap = (self.WORKDAY[m] - count_horizontal[m]) * (self.WORKDAY[m] - count_horizontal[m])
            pena_workday += gap

        if pena_dist:
            pena_dist = round(pena_dist, 1)

        ret = [("0", pena_desire[0]), ("1", pena_desire[1]), ("2", pena_desire[2]), ("3", pena_desire[3]),
               ("昼夜連勤", pena_seq), ("人数超過", pena_over), ("人数不足", pena_lack),
               ("ばらけ具合", pena_dist), ("勤務日数希望違反", pena_workday),
               ("最大連勤", max_seq_work), ("最大連休", max_seq_off)]

        return ret


channel_access_token = "xGrW44hCkpMrzQ58fWVS3ZPAHEA+z7UOLHikUvMO6u592F1F+aTcxKURKx3+CFTT5nu/TTVlzV/I1XlRYiR6lrY7TReIRfLd9AARkClP7CIY5HEezWECcxApeXveX9cuh2RV2Vjqq8P5zeNEjah9XgdB04t89/1O/w1cDnyilFU="
line_bot_api = LineBotApi(channel_access_token)


def main(d):
    model = ShiftAnneal()
    model.setLIST(data_list=d)
    model.setParam(des_const=1000,
                   seq_const=1000,
                   shift_size_const=100,
                   workday_const=1,
                   shift_size_limit=[1 for _ in range(62)],
                   workday=[30, 30],
                   num_reads=5)
    model.setConst()
    model.sample()
    print(model.sample_set.record[model.order][0])

    user_id = "Ufa2bdd3e5bad5382d047a1c23c22cf71"
    line_bot_api.push_message(user_id, messages=TextSendMessage(str(model.sample_set.record[model.order][0])))