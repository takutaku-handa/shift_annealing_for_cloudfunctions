import csv
import math
import numpy as np
import dimod
import time
from neal import SimulatedAnnealingSampler
from google.cloud import storage
from linebot import LineBotApi
from linebot.models import TextSendMessage, ImageSendMessage
import matplotlib.pyplot as plt

channel_access_token = "xGrW44hCkpMrzQ58fWVS3ZPAHEA+z7UOLHikUvMO6u592F1F+aTcxKURKx3+CFTT5nu/TTVlzV/I1XlRYiR6lrY7TReIRfLd9AARkClP7CIY5HEezWECcxApeXveX9cuh2RV2Vjqq8P5zeNEjah9XgdB04t89/1O/w1cDnyilFU="
user_id = "Ufa2bdd3e5bad5382d047a1c23c22cf71"
bucket_name = "shift_sa"


def hello_world(request):
    request_json = request.get_json()
    if request_json and "messages" in request_json:
        print(request_json["messages"])
        main(request_json["messages"])
        return "Got List!"
    else:
        print("No List!")
        return "No List"


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

        ret = [("Lv0", pena_desire[0]), ("Lv1", pena_desire[1]), ("Lv2", pena_desire[2]), ("Lv3", pena_desire[3]),
               ("L/D", pena_seq), ("Over", pena_over), ("Lack", pena_lack),
               ("Dist", pena_dist), ("Work", pena_workday),
               ("MaxSeq", max_seq_work), ("MaxOff", max_seq_off)]

        return ret


def upload_img(fn):
    source_file = "/tmp/{0}".format(fn)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(fn)
    blob.upload_from_filename(source_file)
    blob.make_public()
    return blob.public_url


def create_overview_img(model, size: int):
    keys = []
    for key in model.getPenalty(model.sample_set.record[model.order][0][0]):
        keys.append(key[0])
    sample_list = model.sample_set.record[model.order][:size]
    data = []
    for sample in sample_list:
        penalty = model.getPenalty(sample[0])
        tmp = []
        for p in penalty:
            tmp.append(p[1])
        data.append(tmp)

    fig = plt.figure(figsize=(5, 3), dpi=240)
    ax1 = fig.add_subplot(111)
    ax1.axis('off')
    ax1.table(cellText=np.array(data).T.astype(int), colLabels=[i for i in range(1, size + 1)],
              rowLabels=keys, loc="center")
    fig.tight_layout()
    fn = "overview_{0}.jpg".format(time.time())
    plt.savefig("/tmp/{0}".format(fn))
    plt.close()

    url = upload_img(fn)
    return url


def main(d):
    model = ShiftAnneal()
    model.setLIST(data_list=d)
    model.setParam(des_const=1000,  # -----------------------------------
                   seq_const=1000,  # -----------------------------------
                   shift_size_const=100,  # -----------------------------
                   workday_const=1,  # ----------------------------------
                   shift_size_limit=[1 for _ in range(62)],  # ----------
                   workday=["ignore"],  # setLISTで設定しているから使わない。
                   num_reads=10)  # -------------------------------------
    model.setConst()
    model.sample()
    first = model.sample_set.record[model.order][0]
    print(first)
    print(model.getPenalty(sample=first[0]))

    line_bot_api = LineBotApi(channel_access_token)
    line_bot_api.push_message(user_id, messages=TextSendMessage(str(model.sample_set.record[model.order][0])))

    url_overview = create_overview_img(model, 8)
    line_bot_api.push_message(user_id, messages=ImageSendMessage(original_content_url=url_overview,
                                                                 preview_image_url=url_overview))


if __name__ == "__main__":
    main(d=[["bob", 0, 1, 2, 0, 2, 1, 3], ["tom", 1, 0, 0, 1, 0, 2, 3]])
