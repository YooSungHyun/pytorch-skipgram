import math
import pandas
from utils.abc_dataset import CustomDataset
import numpy
from collections.abc import Callable
from collections import defaultdict, Counter
import copy


class CBOWDataset(CustomDataset):
    def __init__(self, df: numpy.ndarray, counter: dict = None, transform: Callable = None):
        self.df = df
        self.counter = Counter(counter)
        if self.counter:
            total_word = self.counter.total()
            self.subsample_freq = copy.deepcopy(self.counter)
            self.negative_freq = copy.deepcopy(self.counter)

            """
            만일 𝑓(𝑤𝑖)가 0.01로 나타나는 빈도 높은 단어('은/는')는 위 식으로 계산한 𝑃(𝑤𝑖)가 0.9684나 되어서 100번의 학습 기회 가운데
            96번 정도는 학습에서 제외하게 됩니다. 반대로 등장 비율이 적어 𝑃(𝑤𝑖)가 0에 가깝다면 해당 단어가 나올 때마다
            빼놓지 않고 학습을 시키는 구조입니다.
            """
            total_freq_vi = 0
            for key, val in self.subsample_freq.items():
                freq_vi = val / total_word
                total_freq_vi += freq_vi
                P_vi = 1 - math.sqrt(10**-5 / freq_vi)
                self.subsample_freq[key] = P_vi

            for key, val in self.negative_freq.items():
                freq_vi = val / total_word
                self.negative_freq[key] = (freq_vi ** (3 / 4)) / total_freq_vi ** (3 / 4)
            self.negative_list = sorted(self.negative_freq.items(), key=lambda x: x[1])
        self.transform = transform

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = list()
        if isinstance(self.df.iloc[idx], pandas.core.series.Series):
            data.append(dict(self.df.iloc[idx]))
        else:
            for _, series in self.df.iloc[idx].iterrows():
                data.append(dict(series))

        result_dict = defaultdict(list)
        if isinstance(idx, (slice, tuple, list)):
            for i in range(len(data)):
                # set_transform effect
                if self.transform:
                    for key, val in self.transform(data[i], self.subsample_freq, self.negative_list).items():
                        result_dict[key].append(val)
        else:
            # set_transform effect
            if self.transform:
                for key, val in self.transform(data[-1], self.subsample_freq, self.negative_list).items():
                    result_dict[key].append(val)

        return dict(result_dict)
