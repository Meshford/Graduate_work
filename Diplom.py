#!/usr/bin/env python
# coding: utf-8

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import sklearn
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import preprocessing
import json
from IPython.display import display
import tensorflow as tf
from tensorflow import keras

from statistics import mean
from math import sqrt
pd.options.mode.chained_assignment = None


def main():
    ch_names = ['VentricularRate', 'AtrialRate', 'QRSDuration', 'QTInterval', 'QTCorrected', 'RAxis', 'TAxis',
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
        record_data = json.load(open(path_to_data + record.FileName + ".json", 'r'))
        VentricularRate = []
        AtrialRate = []
        QRSDuration = []  # duration qrs in ms
        QTInterval = []  # numeric Interval between start of QRS and end of T wave, ms
        QTTCorrected = []  # numeric Corrected QT-interval according to Bazzet's formula, ms
        RAxis = []
        TAxis = []
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
        for i in range(0, len(QTInterval)):
            if RR_interval[i] == 0:
                RR_interval[i] = mean(RR_interval)
            QTTCorrected.append(QTInterval[i] / sqrt(RR_interval[i]))
        QRSCount.append(count_q)
        AtrialRate.append(count_p * 6)
        VentricularRate.append(int((len(R_wave)-1) * 6))
        macro_data.loc[macro_data.index[i], 'New_QRSCount'] = QRSCount
        macro_data.loc[macro_data.index[i], 'New_QOnset'] = QOnset
        macro_data.loc[macro_data.index[i], 'New_QOffset'] = QOffset
        macro_data.loc[macro_data.index[i], 'New_TOffset'] = TOffset
        macro_data.loc[macro_data.index[i], 'New_QTInterval'] = int(mean(QTInterval) / sample_rate) * 1000
        macro_data.loc[macro_data.index[i], 'New_QRSDuration'] = int(mean(QRSDuration) / sample_rate) * 1000
        macro_data.loc[macro_data.index[i], 'QTTCorrected'] = int(mean(QTTCorrected) / sample_rate) * 1000
        macro_data.loc[macro_data.index[i], 'VentricularRate'] = VentricularRate
        macro_data.loc[macro_data.index[i], 'AtrialRate'] = AtrialRate
    print(macro_data.head(5))

    # Теперь добавим диагнозы из нашего изначального датасета.
    # macro_data["Rhythm"] = data["Rhythm"]


if __name__ == "__main__":
    main()