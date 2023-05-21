import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import pyplot as plt
import os
import numpy as np
from sklearn.metrics import precision_score
import math

# '''
# thr:閾値
# anorm: 異常度が格納されている配列
# test_v: 正解ラベルがついているテストデータ 'label'カラムに正解ラベルが格納されている
# '''
def validate(test_v, anorm, thr):
    test_v['z']=np.where(anorm>=thr, 1, 0) #'z'カラムに予測値を格納する
    test_v.reset_index(inplace=True, drop=True)
    tp=test_v[((test_v['label']==1)|(test_v['label']==2))&(test_v['z']==1)] #tp: TPのインデックス
    z_p=test_v[test_v['z']==1] #予測値がPositiiveのインデックス
    pre_score=len(tp)/len(z_p)

    #     再現率（）
    df_anorm=[] #異常音の帯（異常音の範囲）（データフレーム）を集めるリスト
    search= 1 if test_v.loc[0, 'label']==0 else 0 #なんのラベルを探すか（searchが1のときはラベルが1(または2)の行を探す）
    start = 0
    for num in range(len(test_v)):
        if search==1 and ((test_v.loc[num, 'label']==1)or(test_v.loc[num, 'label']==2)): #学習データのラベルが1または2のとき異常
            start=num
            search=0
        elif search==0 and test_v.loc[num, 'label']==search:
            stop=num-1
            anorm_range=test_v.loc[start:stop].copy() #異常音の範囲のデータフレーム
            df_anorm.append(anorm_range)
            search=1

    if start>stop:
        anorm_range=test_v.loc[start:].copy()
        df_anorm.append(anorm_range)    
    
    num_positive = 1 # 異常音の帯の中の何点以上を「異常音」と予測した時にその異常音の帯を異常音として判断したことにするか
    count=[] #異常音の帯の中に異常と判断した点がnum_positive以上ある異常音の帯を格納するリスト
    for i in range(len(df_anorm)):
        if len(df_anorm[i].loc[df_anorm[i]['z']==1])>=num_positive: #異常音の帯の中に異常と判断した点がnum_positive以上の場合:
               count.append(i)    
#     print(len(df_anorm))
    re_score=len(count)/len(df_anorm)

    print('適合率：%f'%(pre_score*100))    
    print('再現率：%f'%(re_score*100))
    
    return pre_score, re_score

# 等価騒音レベルを求める関数
def equivalentSoundLevel(x):
#     '''
#     x:騒音データ(ndarray)
#     '''
    t = len(x)
    y = np.zeros(t)
    for i in range(t):
        y[i] = 10**(x[i]/10)
    mean_y = np.mean(y)
    Laeq =10*np.log10(mean_y)
    
    return Laeq


def figure(df_test, anorm=[]):
    test_plot=df_test['original'].values
    num_ax=math.floor(len(test_plot)/17999)
    label=df_test['label'].values*100
    label_index=range(len(label))

    fig, ax=plt.subplots(num_ax, 1, figsize=(35, 15*num_ax))
    for i in range(num_ax):
        ax[i].plot(anorm, '-r',linewidth = 1 )
        ax[i].plot(test_plot, '-k',linewidth = 2)
        ax[i].fill_between(label_index, label, facecolor='lime' )
        ax[i].set_ylim(0, 90)
        ax[i].set_xlim(i*17999, 17999*(i+1))
    plt.show()

