#!/usr/bin/env python
# coding: utf-8

import torch
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import json
from IPython.display import display
import tensorflow as tf
from tensorflow import keras
from random import randrange
from statistics import mean
from math import sqrt
pd.options.mode.chained_assignment = None
import wfdb

sample_rate = 500
path_to_model = "Segmentation/"
path_to_data = "data/"


class Delineation:
    def __init__(self):
        self._model = torch.jit.load(path_to_model + "model.pt")
        self._max_dist = int(0.03 * sample_rate)
        self._border = int(0.8 * sample_rate)
    #удаляет маленькие волны
    def _remove_small(self, signal):
        last_zero = 0
        for i in range(len(signal)):
            if signal[i] == 0:
                if i - last_zero < self._max_dist:
                    signal[last_zero:i] = 0
                last_zero = i
    #мерджит две волны одного типа
    def _merge_small(self, signal):
        lasts = np.full(signal.max() + 1, -(self._max_dist + 1))
        for i in range(len(signal)):
            m = signal[i]
            if i - lasts[m] < self._max_dist and m > 0:
                signal[lasts[m]:i] = m
            lasts[m] = i

    def _mask_to_delineation(self, data):
        masks = np.argmax(data, 1)
        delineation = []
        v_to_del = {0: 'none', 1: 'p', 2: 'qrs', 3: 't'}
        for rec in masks:
            self._merge_small(rec)
            self._remove_small(rec)
            rec_del = []
            i = 0
            rec_len = len(rec)
            while i < rec_len:
                v = rec[i]
                if v > 0:
                    rec_del.append({
                        "begin": i,
                        "end": 0,
                        "type": v_to_del[v]
                    })
                    while i < rec_len and rec[i] == v:
                        rec_del[-1]["end"] = i
                        i += 1
                    t = rec_del[-1]
                    if t["begin"] < self._border or t["end"] > rec_len - self._border:
                        rec_del.pop()
                i += 1
            d_res = []
            for c, n in zip(rec_del[:-1], rec_del[1:]):
                d_res.append(c)
                d_res.append({
                    "begin": c["end"],
                    "end": n["begin"],
                    "type": "none"
                })
            if rec_del:
                begin = {
                    "begin": 0,
                    "end": rec_del[0]["begin"],
                    "type": "none"
                }
                end = {
                    "begin": rec_del[-1]["end"],
                    "end": rec_len,
                    "type": "none"
                }
                d_res = [begin] + d_res + [rec_del[-1], end]
            else:
                d_res.append({
                    "begin": 0,
                    "end": rec_len,
                    "type": "none"
                })
            delineation.append(d_res)
        return delineation

    def __call__(self, signal):
        signal = torch.FloatTensor(np.expand_dims(signal, axis=1))
        masks = self._model(signal).data.numpy()
        return self._mask_to_delineation(masks)


def call_deleniation():
    database_info = pd.read_csv(path_to_data + "Diagnostics.csv", delimiter=';')
    delineation = Delineation()
    i = 0
    for f in database_info['FileName']:
        signal = pd.read_csv("ECGDataDenoised/" + f + '.csv', delimiter=',', header=None)
        data = np.array(signal) / 1000
        data_transp = np.transpose(data)
        result = delineation(data_transp)
        with open(path_to_data + f + ".json", "w") as write_file:
            json.dump(result, write_file)
        print(i, " record passed")
        i += 1
#
# def plot_signal_with_mask(signal):
#     plt.figure(figsize=(18, 5))
#     plt.title("Сигнал с маской")
#     plt.xlabel("Дискретная величина")
#     plt.ylabel("Амплитуда (мВ)")
#     x_axis_values = np.linspace(0, 5000, len(signal))
#     plt.plot(x_axis_values, signal, linewidth=2, color="black")
#     plt.xticks(np.arange(0, 5001, 200))
#     plt.show()

def main():

    #call_deleniation()
    ch_names = ['VentricularRate', 'AtrialRate', 'QRSDuration', 'QTInterval', 'QTCorrected',
                'QRSCount', 'QOnset', 'QOffset', 'TOffset']
    attributes = ["Gender", "PatientAge"]
    for ch_names in ch_names:
        attributes.append('New_' + ch_names)
    macro_data = pd.DataFrame(columns=attributes)

    # Добавим из нашего датасета, с которым работали в начале, в наш новый датасет возраст и пол пациента
    data = pd.read_csv("data/Diagnostics.csv", delimiter=';')
    macro_data["Gender"] = data["Gender"]
    macro_data["PatientAge"] = data["PatientAge"]

    sample_rate = 500
    path_to_model = "Segmentation/"
    path_to_data = "data/"
    for i, record in enumerate(data.iloc):
        # итерируемся по 1 объекту
        #signal = pd.read_csv("ECGDataDenoised/" + record.FileName + '.csv', delimiter=',', header=None)
        #plot_signal_with_mask(signal[1])
        record_data = json.load(open(path_to_data + record.FileName + ".json", 'r'))
        VentricularRate = []
        AtrialRate = []
        QRSDuration = []  # duration qrs in ms
        QTInterval = []  # numeric Interval between start of QRS and end of T wave, ms
        QTTCorrected = []  # numeric Corrected QT-interval according to Bazzet's formula, ms
        QRSCount = []
        QOnset = []
        QOffset = []
        TOffset = []
        # то есть мы будем итерироваться только по 2 отведению ?
        #for ch_num, chanel in enumerate(record_data):
            # итерируемся по 12 отведениям
            # списки длин соответствующих волн, комплексов, интервалов
        chanel = record_data[1]  # указываем второе отведение
        count_q = 0
        count_t = 0
        count_p = 0
        R_wave = [0]
        RR_interval = []
        for wave_num, wave in enumerate(chanel):
            # итерируемся по сигналу 1-го отведения
            if wave['type'] == 'p':
                count_p += 1
            if wave['type'] == 'qrs':
                R_wave.append(mean([wave['begin'], wave['end']]))
                RR_interval.append(R_wave[-1] - R_wave[-2])
                count_q += 1
                if count_q == 1:
                    QOnset.append(wave['begin'])
                    QOffset.append(wave['end'])
                # Собираем длины qrs комплексов в список
                QRSDuration.append(wave['end'] - wave['begin'])
                if wave_num < len(chanel) - 2:
                    if (chanel[wave_num + 1]['type'] == 't'):
                        QTInterval.append(chanel[wave_num + 1]['end'] - wave['begin'])
                    if (chanel[wave_num + 2]['type'] == 't'):
                        QTInterval.append(chanel[wave_num + 2]['end'] - wave['begin'])
            if wave['type'] == 't':
                count_t += 1
                if count_t == 1:
                    TOffset.append(wave['end'])
        if len(QTInterval) == 0:
            QTInterval.append(0)
            RR_interval.append(1)
            QTTCorrected.append(0)
        else:
            for j in range(0, len(QTInterval)):
                if RR_interval[j] == 0:
                    RR_interval[j] = mean(RR_interval)
                QTTCorrected.append(QTInterval[j] / sqrt(RR_interval[j]))
        if count_t == 0:
            TOffset.append(0) #есть сигнал плохо сегментируемый MUSE_20180114_075129_97000
        if count_q == 0:
            QOnset.append(0) #MUSE_20180712_151203_59000
            QOffset.append(0) #MUSE_20180712_151203_59000
            QRSDuration.append(0) #MUSE_20180712_151203_59000
        count_q += 2 # костыль тк сеть режет конец и начало
        QRSCount.append(count_q)
        if count_p < 5:
            AtrialRate.append(randrange(150, 400))
        else:
            AtrialRate.append(int(count_p * 6))
        #VentricularRate.append(int((len(R_wave)-1) * 6))
        VentricularRate.append(int(count_q * 6))
        macro_data.loc[macro_data.index[i], 'New_QRSCount'] = QRSCount
        macro_data.loc[macro_data.index[i], 'New_QOnset'] = QOnset
        macro_data.loc[macro_data.index[i], 'New_QOffset'] = QOffset
        macro_data.loc[macro_data.index[i], 'New_TOffset'] = TOffset
        macro_data.loc[macro_data.index[i], 'New_QTInterval'] = int(mean(QTInterval) / sample_rate * 1000)
        macro_data.loc[macro_data.index[i], 'New_QRSDuration'] = int(mean(QRSDuration) / sample_rate * 1000)
        macro_data.loc[macro_data.index[i], 'New_QTCorrected'] = int(mean(QTTCorrected) / sample_rate * 1000)
        macro_data.loc[macro_data.index[i], 'New_VentricularRate'] = VentricularRate
        macro_data.loc[macro_data.index[i], 'New_AtrialRate'] = AtrialRate
    #print(macro_data)

    #Теперь добавим диагнозы из нашего изначального датасета.
    macro_data.drop(macro_data.index[10606:10645], inplace=True)
    macro_data["Rhythm"] = data["Rhythm"]
    #удалить все что свыше 10608
if __name__ == "__main__":
    main()

