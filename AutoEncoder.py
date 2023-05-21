import numpy as np
import pandas as pd
import librosa as lb
import os
import h5py
import joblib #並列処理のためのライブラリ
import shutil
from tqdm import tqdm 
import matplotlib.pyplot as plt
import glob
import wave
import gc
from tensorflow.keras.utils import Sequence
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import models
from tensorflow import keras 
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback
from memory_profiler import profile

class genSequence(Sequence):

    def __init__(self, data_path, data_size, key_name, batch_size):
        self.data_path = data_path
        self.data_size = data_size
        self.batch_size = batch_size
        self.key_name = key_name

    def __len__(self):
        return int(np.ceil(self.data_size / float(self.batch_size)))

    def __getitem__(self, idx):
        with h5py.File(self.data_path, 'r') as f:
            batch_x = f[self.key_name][idx * self.batch_size:(idx + 1) * self.batch_size]

#         batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
#         batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
#         batch_x =self.preprocess(batch_x) # 前処理

        return batch_x, batch_x
    
#     def preprocess(self, image):
#         # いろいろ前処理
#         return image

    def on_epoch_end(self):
        tf.keras.backend.clear_session() # 計算グラフを破棄する
        gc.collect() #回収可能なオブジェクトを削除
        
# エポックごとにメモリを整理する
class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):     
        tf.keras.backend.clear_session() # 計算グラフを破棄する
        gc.collect() #回収可能なオブジェクトを削除


#正常音(または異常音)だけを切り抜いてメルスペクトログラムを作成しnpy形式に保存するためのクラス
class Sound_load_and_save():
    def __init__(self, sr, duration, melparams, way):
        '''
        sr:スペクトログラムに変換する音データのサンプリング周波数，すべてのデータのsrが揃っていない場合は揃える
        duration:メルスペクトログラムの時間幅
        melparams:メルスペクトログラムの各種パラメータの辞書
        way:切り抜きの仕方．
            True:正常音の部分を切り抜く
            False: 異常音の部分を切り抜く
        '''
        self.sr = sr
        self.duration = duration
        self.melparams = melparams   
        self.way = way
        
    # 正常尾を抜き出す
    def clipping_normal(self, label):
        '''
        label:1列の正解ラベルのデータフレーム(ラベルの列だけ)，0:正常, 1:異常
        戻り値:
            list_start:正常音の開始時間を格納したリスト
            list_duration:正常音の継続時間を格納したリスト

        '''
        list_start = [] # 正常音の開始時間(s)
        list_duration = []# 正常音の時間長(s)
        search= 1 if label.iloc[0, 0]==0 else 0 #なんのラベルを探すか（searchが1のときはラベルが1の行を探す）
        start = 0
        for num in range(len(label)):
            if search==0 and (label.iloc[num, 0]==search): 
                start=num
                search=1
            elif search==1 and label.iloc[num, 0]==search:
                stop=num-1
                duration = (stop-start+1)*0.2
                list_start.append(start*0.2)
                list_duration.append(duration)
                search=0
        
        if start>stop:
            stop=len(label)-1
            duration = (stop-start+1)*0.2
            list_start.append(start*0.2)
            list_duration.append(duration)

        return list_start, list_duration


    # 異常音を抜き出す
    def clipping_abnorm(self, label):
        '''
        label:1列の正解ラベルのデータフレーム(ラベルの列だけ)，0:正常, 1:異常
        戻り値:
            list_start:異常音の開始時間を格納したリスト
            list_duration:異常音の継続時間を格納したリスト

        '''
        list_start = [] # 異常音の開始時間(s)
        list_duration = []# 異常音の時間長(s)
        search= 1 if label.iloc[0, 0]==0 else 0 #なんのラベルを探すか（searchが1のときはラベルが1の行を探す）
        start = 0
        for num in range(len(label)):
            if search==1 and (label.iloc[num, 0]==search): 
                start=num
                search=0
            elif search==0 and label.iloc[num, 0]==search:
                stop=num-1
                duration = (stop-start+1)*0.2
                list_start.append(start*0.2)
                list_duration.append(duration)
                search=1
 
        if start>stop:
            stop=len(label)-1
            duration = (stop-start+1)*0.2
            list_start.append(start*0.2)
            list_duration.append(duration)

        return list_start, list_duration


    # メルスペクトログラム変換の関数
    # melparams: パラメータを格納した辞書型変数
    def compute_melspec(self, y, melparams):
        melspec = lb.power_to_db(lb.feature.melspectrogram(y, **melparams)).astype(np.float32)
        return melspec

    # record:音データのパス
    def load_and_save(self, record, out_dir, transform, label_data_df):
        '''
        record:音声ファイルのパス
        out_dir:npyファイル出力するディレクトリのパス
        transform: audio_DA.pyにあるデータ拡張クラスをまとめたオブジェクト
        label_data_df:正解ラベルのデータフレーム
        '''
        sr = self.sr
        duration = self.duration
        melparams = self.melparams
        way = self.way

        # ラベルデータをもとに正常音の部分だけを取り出す
        if way:
            list_start, list_duration = self.clipping_normal(label_data_df)
        else:
            list_start, list_duration = self.clipping_abnorm(label_data_df)

        count = 0 # ファイル名を付けるときに使用
        for offset, length in tqdm(zip(list_start, list_duration)):
            # 分割後，何個のフレームができるか
            num_cut = int(length//duration)
            for i in tqdm(range(num_cut)):       
                y, sr = lb.load(record, sr = None, offset=offset+(i*duration), duration=duration)
                y_augmented = transform(y) # データ拡張        
                melspec = self.compute_melspec(y, melparams)
                melspec_augmented = self.compute_melspec(y_augmented, melparams)       

                #ファイル名の最初の文字列に、処理前のデータに”0”をつけ処理後は”1”をつけて区別する
                record_name = '0'+'_'+str(count)+ '_' +record.split('/')[-1].replace('.wav', '.npy') 
                augmented_record_name = '1'+record_name.replace('.wav', '.npy')

                np.save(f'{out_dir}/{record_name}', melspec)
                np.save(f'{out_dir}/{augmented_record_name}', melspec_augmented)

                count+=1
                
                
# スペクトログラムを3チャンネルにして，HDF5ファイルに変換            
class save_as_hdf5:       
    
    def __init__(self, new_hdf, npy_files_path):
        """
        new_hdf: 「.hdf5」形式ファイルのパス
        npy_files_path:hdfに変換するnpyファイルがあるフォルダのパス
        """
        self.new_hdf = new_hdf
        self.npy_files_path = npy_files_path
        
        
    '''
    スペクトルのスケールを0～255に変換し，カラーに変換．
    同じチャンネルを複製して３チャンネルにしているだけなので，見た目はグレースケールとほぼ同じ
    '''
    # 正規化
    def MonoToColor(self, X, eps=1e-6, mean=None, std=None):
        mean = mean or X.mean() 
        std = std or X.std()
        Xstd = (X - mean) / (std + eps) # epsでゼロ除算を防ぐ
    #     print(type(eps))

        _min, _max = Xstd.min(), Xstd.max()

        if (_max - _min) > eps:
            V = np.clip(Xstd, _min, _max)  
            V = 255 * (V - _min) / (_max - _min)
            V = V.astype(np.uint8)
        else: 
            V = np.zeros_like(Xstd, dtype=np.uint8) 

        V = np.stack([V, V, V], axis=-1)
        V = V.astype('float32')/255 
        return V

    def __call__(self):
        new_hdf = self.new_hdf
        npy_files_path = self.npy_files_path
        
        with h5py.File(new_hdf, mode='w') as f:  
            train_val_files = glob.glob(f'{npy_files_path}*')
            print('全データ数:', len(train_val_files))

            np.random.seed(seed=32)
            val_rate = 0.1 # 訓練データの何割をバリデーションデータに割り当てるか
            index_array = np.array(range(len(train_val_files)))
            np.random.shuffle(index_array)
            train_data_num = int(len(train_val_files)*(1-val_rate))
#             print('全データ×(1-val_rate):', train_data_num)

            # train_val_filesを訓練データとバリデーションデータに分ける
            train_files=[]
            val_files=[]
            for i, idx in enumerate(index_array):
                if i < train_data_num:
                    train_files.append(train_val_files[idx])
                else:
                    val_files.append(train_val_files[idx])
            print('訓練データ数：',len(train_files))
            print('valデータ数：',len(val_files))

            mean = []
            std = []
            # 各データのスペクトログラムの平均値をmeanに格納し，それをさらに平均したものをmeanに格納する
            # すべてのデータのスペクトログラムの平均値と標準偏差を使って正規化するため
            for i in tqdm(train_val_files):
                file = np.load(i)
                mean.append(file.mean())
                std.append(file.std())

            mean = np.array(mean).mean() 
            std = np.array(std).mean()     

            base = np.load(train_files[1])
            val_base = np.load(val_files[1])
            print('train_base:', base.shape)
            print('val_base:', val_base.shape)

            train_shape = (len(train_files), *base.shape, 3)
            val_shape = (len(val_files), *base.shape, 3)
            print('train_shape', train_shape) # (データ数，スペクトログラムの行数，スペクトログラムの列数, チャンネル数）
            print('val_shape', val_shape)

        #     f.create_dataset('train_files', train_shape, dtype = np.uint8) # np.uint8:符号なし8ビット整数型
        #     f.create_dataset('val_files', val_shape, dtype = np.uint8) 

            f.create_dataset('train_files', train_shape, dtype = np.float32) # MonoToColorで0-1に正規化する場合
            f.create_dataset('val_files', val_shape, dtype = np.float32) 
            # 学習データの平均値と標準偏差を保存
            f.create_dataset('mean', data=mean) 
            f.create_dataset('std', data=std) 

            f.create_dataset('train_labels', (len(train_files),), dtype = 'S50')
            f.create_dataset('val_labels', (len(val_files),), dtype = 'S50')

            f['train_labels'][...] = [i.split('\\')[-1].split('.')[0].encode(encoding="ascii", errors = "ignore") for i in train_files]
            f['val_labels'][...] = [i.split('\\')[-1].split('.')[0].encode(encoding="ascii", errors = "ignore") for i in val_files]

            for i, v in tqdm(enumerate(train_files), total=len(train_files)):
                f['train_files'][i, ...] = self.MonoToColor(np.load(v), mean=mean, std=std)

            for i, v in tqdm(enumerate(val_files), total=len(val_files)):
                f['val_files'][i, ...] = self.MonoToColor(np.load(v), mean=mean, std=std)
        return mean, std
                

# オートエンコーダの出力から作成した正解ラベルを元のラベルに追加する
class AddLabel():
    def __init__(self, original_label_data, duration, thr, way):
        '''
        original_label_data:ラベルを追加する前の正解ラベルの配列(ndarray)
        duration: スペクトログラムの時間幅[s]
        thr:閾値
        way:異常ラベルの追加の仕方．
            True:元のラベルに加えて異常度が閾値よりも大きいところに異常ラベルを追加する
            False: 異常度が閾値よりも大きい，かつ，元のラベルが異常のところのみに異常ラベルをつける
        '''
        self.original_label_data = original_label_data
        self.duration = duration
        self.thr = thr
        self.way = way

    # バッチごとに予測し異常度を算出
    def predict_batch_add_label(self, data, batch_size, model):   
        '''
        data: ラベルを追加したい日付のデータ（スペクトログラム）
        batch_size:予測する際のバッチサイズ
        model:予測につかう学習済みモデル        

        '''
        predict_batch_size = batch_size
         
        decoded_imgs = np.empty(data.shape, dtype=np.float32)  

        BATCH_INDICES = np.arange(start=0, stop=len(data), step=predict_batch_size)  #バッチのインデックス
        BATCH_INDICES = np.append(BATCH_INDICES, len(data))  # 最後のバッチのインデックスを追加

        for index in tqdm(np.arange(len(BATCH_INDICES) - 1)):
            batch_start = BATCH_INDICES[index]
            batch_end = BATCH_INDICES[index + 1]
            decoded_imgs[batch_start:batch_end] = model.predict_on_batch(data[batch_start:batch_end])
        
        # 異常度(平均2乗誤差)算出
        print('decoded_imgs.shape:',decoded_imgs.shape)
        print('data.shape:',data.shape)
        anomary_scores = np.zeros(len(decoded_imgs))
        for i in range(len(anomary_scores)):
            anomary_scores[i] = np.mean((decoded_imgs[i]-data[i])**2)
        print('anomary_scores.shape:', anomary_scores.shape)
        
        added_label = self.give_label(anomary_scores)

        return added_label


    # オートエンコーダが出力した異常度の結果から0.2秒単位の正解ラベルの配列を作る
    def give_label(self, result):
        '''
        result:オートエンコーダが出力した異常度の結果(ndarray)
        戻り値:異常ラベルを追加した最終的な正解ラベルの配列(ndarray)
        '''
        duration = self.duration
        thr = self.thr
        way = self.way
        
        total_time = len(result)*duration
        interval = int(duration/0.2)
        z_anorm = np.zeros(int(total_time/0.2))
        for i in range(len(result)):
            if i<len(result)-1:
                z_anorm[i*interval:(i+1)*interval] = result[i]
            else:
                z_anorm[i*interval:] = result[i]               

        added_label = self.original_label_data
        # 音声データと騒音レベルデータの長さが異なるので長さをそろえる
        if len(added_label)<=len(z_anorm):
            z_anorm = z_anorm[:len(self.original_label_data)] 
        else:
            added_label = added_label[:len(z_anorm)]

        if way:
            z_anorm = np.where(z_anorm >= thr, 1, 0)
            added_label = np.where(z_anorm == 1, 1, added_label) # 元のラベルに加えて異常度が閾値よりも大きいところに異常ラベルを追加する
        else:
            z_anorm = np.where(z_anorm >= thr, 1, 0)
            added_label = np.where((z_anorm == 1)&(added_label==1), 1, 0) # 作成した教師ラベルが1，かつ，元のラベルが1,なら1，それ以外なら0

        return added_label       
    
class FindThreshold:
    def __init__(self, data_path, label, model, batch_size, multi):
        '''
        data_path:予測に使うhdf5ファイルのパス
        multi : 四分位範囲を何倍するか
        '''
        self.data_path = data_path
        self.label = label
        self.model = model
        self.batch_size = batch_size
        self.multi = multi
            
    # 外れ値除去
    def outlier_iqr(self, df):
        for i in range(len(df.columns)):
            # 列を抽出する
            col = df.iloc[:,i]

            # 四分位数
            q1 = col.describe()['25%']
            q3 = col.describe()['75%']
            iqr = q3 - q1 #四分位範囲

            # 外れ値の基準点
            outlier_min = q1 - (iqr) * self.multi
            outlier_max = q3 + (iqr) * self.multi

            # 範囲から外れている値を除く
            col[col < outlier_min] = None
            col[col > outlier_max] = None

        return df

    def __call__(self):
        '''
        戻り値:label（正常音，または異常音)で指定したデータに対してmodelが出力した異常度の（最小値，最大値)のタプル
        '''
        data_path = self.data_path
        label = self.label
        model = self.model
        predict_batch_size = self.batch_size
        with h5py.File(self.data_path, 'r') as f:
            shape = f[f'{label}_files'].shape
            length = len(f[f'{label}_files'])
            print('len(h5py.File(data_path, 'r')[{label}_files]):', len(f[f'{label}_files']))

            decoded_imgs = np.empty(shape, dtype=np.float32)  
            BATCH_INDICES = np.arange(start=0, stop=length, step=predict_batch_size) #バッチのインデックス
            BATCH_INDICES = np.append(BATCH_INDICES, length)  # 最後のバッチのインデックスを追加

            for index in tqdm(np.arange(len(BATCH_INDICES) - 1)):
                batch_start = BATCH_INDICES[index]  
                batch_end = BATCH_INDICES[index + 1]
                decoded_imgs[batch_start:batch_end] = model.predict_on_batch(f[f'{label}_files'][batch_start:batch_end])

            # 異常度(平均2乗誤差)算出
            print('decoded_imgs.shape:',decoded_imgs.shape)
            print('data.shape:',f[f'{label}_files'].shape)
            anomary_scores = np.zeros(len(decoded_imgs))
            for i in range(len(anomary_scores)):
                anomary_scores[i] = np.mean((decoded_imgs[i]-f[f'{label}_files'][i])**2)
            print('anomary_scores.shape:', anomary_scores.shape)

        # 外れ値除去
        anomary_scores_df = pd.DataFrame(anomary_scores)
        anomary_scores_df = self.outlier_iqr(anomary_scores_df)
        anomary_scores_df = anomary_scores_df.dropna(how='any', axis=0)

        return anomary_scores_df[0].min(), anomary_scores_df[0].max()
    
#テストデータをメルスペクトログラムに変換しhdf5形式で保存
class TestSaveAsHdf5:
    def __init__(self, wav_path_in, duration, melparams):
        '''
        wav_path_in :testdataのwavファイルのパス
        duration :スペクトログラム時間幅[s]
        melparams:メルスペクトログラムのパラメータを格納した辞書
        hdf5_out_dir :hdf5の出力先
        戻り値：テストデータのhdf5の出力先のパス
        '''        
        self.wav_path_in = wav_path_in
        self.duration = duration
        self.melparams = melparams
        self.basename = os.path.splitext(os.path.basename(self.wav_path_in))[0]
        self.new_hdf = './data_output/hdf5/testdata_'+self.basename+'.hdf5'
        if os.path.exists(self.new_hdf):
            print(f'{self.new_hdf}が既に存在するため削除します')
            os.remove(self.new_hdf)
            print(f'{self.new_hdf}削除完了')        
        
        self.npy_output = './data_output/npy_files/test/'
        if(os.path.isdir(self.npy_output) == True):
            print(f'{self.npy_output}が既に存在するため削除します')
            shutil.rmtree(self.npy_output)
            print(f'{self.npy_output}削除完了')
        os.makedirs(self.npy_output, exist_ok=False)
        
    # 正規化
    def MonoToColor(self, X, eps=1e-6, mean=None, std=None):
        mean = mean or X.mean() 
        std = std or X.std()
        Xstd = (X - mean) / (std + eps) # epsでゼロ除算を防ぐ
    #     print(type(eps))

        _min, _max = Xstd.min(), Xstd.max()

        if (_max - _min) > eps:
            V = np.clip(Xstd, _min, _max)  
            V = 255 * (V - _min) / (_max - _min)
            V = V.astype(np.uint8)
        else: 
            V = np.zeros_like(Xstd, dtype=np.uint8) 

        V = np.stack([V, V, V], axis=-1)
        V = V.astype('float32')/255 
        return V
        
    def __call__(self, mean, std):
        '''
        mean:正規化に使用する平均値
        std:正規化に使用する標準偏差
        '''        
        new_hdf = self.new_hdf
        file_basename = os.path.splitext(os.path.basename(self.wav_path_in))[0]

        with wave.open(self.wav_path_in) as wav:
            # wavデータのサンプル数を取得
            num_samples = wav.getnframes()
            # サンプリング周波数 [Hz] を取得
            sampling_frequency = wav.getframerate()      
            # 長さ
            total_time = num_samples // sampling_frequency     
            # 分割後，何個のフレームができるか
            num_cut = int(total_time//self.duration)
            print(f'{total_time}[sec]')
            
        # npy形式で保存
        for i in tqdm(range(num_cut)):       
            record_name = str(i)+ '_' + file_basename+'.npy'
            y, sr = lb.load(self.wav_path_in, sr = None, offset=i*self.duration, duration=self.duration)

            melspec = lb.power_to_db(lb.feature.melspectrogram(y=y, **self.melparams)).astype(np.float32)
            np.save(f'{self.npy_output}{record_name}', melspec) # 第一引数にパス文字列、第二引数に保存
            
        with h5py.File(new_hdf, mode='w') as f:  
            file_basename=self.basename
            print(f'file_basename:{file_basename}')
            test_files = glob.glob(f'{self.npy_output}*')
            print('全testデータ数:', len(test_files))  
            
            if (mean==None)or(std==None):
#                 print('meanまたはstdがNoneなので計算します．')
                mean = []
                std = []
                for i in tqdm(test_files):
                    file = np.load(i)
                    mean.append(file.mean())# 各データのスペクトログラムの平均値を求め，meanに追加
                    std.append(file.std())

                mean = np.array(mean).mean() #各スペクトログラムの平均値をさらに平均してをmeanに格納する
                std = np.array(std).mean()                 
            print(f'mean:{mean}')
            print(f'std:{std}')
            base = np.load(test_files[1])
            print('test_base:', base.shape)

            test_shape = (len(test_files), *base.shape, 3)
            print('test_shape', test_shape) # (データ数，スペクトログラムの行数，スペクトログラムの列数, チャンネル数）

            f.create_dataset(f'{file_basename}_files', test_shape, dtype = np.float32) 
            f.create_dataset(f'{file_basename}_labels', (len(test_files),), dtype = 'S50') 

            f[f'{file_basename}_labels'][...] = [i.split('\\')[-1].split('.')[0].encode(encoding="ascii", errors = "ignore") for i in test_files]
            for i, v in tqdm(enumerate(test_files), total=len(test_files)):
                f[f'{file_basename}_files'][i, ...] = self.MonoToColor(np.load(v), mean=mean, std=std)
                  
        if(os.path.isdir(self.npy_output) == True):
            print(f'hdfへの変換が完了したため{self.npy_output}を削除します')
            shutil.rmtree(self.npy_output)
            print(f'{self.npy_output}削除完了')
        
        return self.new_hdf

      
    

class AutoEncoder:
    def __init__(self, each_data_path_group, each_data_original_array_group, threshold_data_path, eachday_hdf, transform, sr, duration, melparams):
        '''
        each_data_path_group       :wavデータのパスを前半と後半に分け，それをリストにしたもの．(形式：((前半のパスのリスト)，(後半のパスのリスト)))
        each_data_original_array_group:前半，後半データの元の正解ラベルの配列をタプルにしたもの(形式：((前半データの正解ラベル配列のリスト), (後半データの正解ラベル配列のリスト)))
        threshold_data_path        :閾値を求める際に使用するデータのパス
        eachday_hdf                :各日のメルスペクトログラムが保存されたhdfファイルのパス
        transform                  :データ拡張用のオブジェクト  
        sr                         :音データのサンプリング周波数
        duration                   :スペクトログラム時間幅[s]
        melparams                  :メルスペクトログラムのパラメータを格納した辞書
        '''
        self.each_data_path_group = each_data_path_group
        self.each_data_original_array_group = each_data_original_array_group     
        self.threshold_data_path = threshold_data_path
        self.eachday_hdf = eachday_hdf
        self.transform = transform
        self.sr = sr
        self.duration = duration
        self.melparams = melparams
        # all_fit()を呼び出した際の全訓練データの正常音のHDFファイルの保存先のパス
        self.alltraindata_hdf_output =  './data_output/hdf5/alltraindata_NormalSound.hdf5'
        # テストデータを正規化する際に使用する訓練データに対する平均値と標準偏差
        self.train_mean = None
        self.train_std = None
        
        #正常音のnpyファイルの出力先をタプルでまとめ，フォルダを作成
        output_firsthalf_train= './data_output/npy_files/正常音/firsthalf_1sec正常音16000hz/'
        output_latterhalf_train= './data_output/npy_files/正常音/latterhalf_1sec正常音16000hz/'
        self.each_data_npy_output = (output_firsthalf_train, output_latterhalf_train)        
        os.makedirs('./data_output/npy_files/正常音/', exist_ok=True)
        
        #正常音のhdf5ファイルの出力先をタプルでまとめ，フォルダを作成       
        firsthalf_normal_hdf = './data_output/hdf5/firsthalf_NormalSound.hdf5'
        latterhalf_normal_hdf = './data_output/hdf5/latterhalf_NormalSound.hdf5'
        self.each_data_hdf_output = (firsthalf_normal_hdf, latterhalf_normal_hdf)      
        os.makedirs('./data_output/hdf5', exist_ok=True)
        
        # 前半，後半それぞれの正常音で学習したモデルの保存先をタプルでまとめ，フォルダを作成
        first_save_model_dir = './data_output/trainedmodel/firsthalf/' # 前半データの正常音で学習させたモデル
        latter_save_model_dir = './data_output/trainedmodel/latterhalf/' # 後半データの正常音で学習させたモデル
        self.each_data_trained_model = (first_save_model_dir, latter_save_model_dir)            
        os.makedirs(first_save_model_dir, exist_ok=True)
        os.makedirs(latter_save_model_dir, exist_ok=True)       
        
        #更新したラベルの保存先を指定して，フォルダを作成
        self.each_data_update_dir = './data_output/label_updated/'
        os.makedirs(self.each_data_update_dir, exist_ok=True)
        
        #更新したラベルの最終結果の保存先を指定して，フォルダを作成
        #コンストラクタを呼び出したときに，以前，実行した結果が上書きされないように最終的な結果と更新中の結果の出力先を分けておく
        self.each_data_final_dir = './data_output/label_final/'
        os.makedirs(self.each_data_final_dir, exist_ok=True)        
  
        # 元の正解ラベル(更新前の正解ラベル)を，更新したラベルの保存先にコピーしておく(ラベル追加の繰り返しの一回目に使うから)
        for data_path_group,  original_array_group in zip(self.each_data_path_group, self.each_data_original_array_group):
            for data_path, original_array in zip(data_path_group, original_array_group):
                label_data_name = data_path.split('/')[-1].replace('.wav', '.csv') 
                save_path = self.each_data_update_dir+label_data_name
                label_data_df = pd.DataFrame(original_array) 
                label_data_df.to_csv(save_path)
                

    # 使用するオートエンコーダの定義
    def make_autoencoder(self, name):
        input_img = keras.Input(shape=(128, 32, 3))

        x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

        x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        decoded = layers.Conv2D(3, (3, 3), activation='sigmoid',padding='same')(x)

        autoencoder = keras.Model(input_img, decoded, name=name)

        return autoencoder
        
    # hdf5からデータを読み込み時系列順に並べなおしたndarrayを返す
    def load_dataset(self, label, h5file):
        with h5py.File(h5file, mode='r') as dataset: 
            shape = dataset[f'{label}_files'].shape
            print(f'dataset[{label}_files].shape:{shape}')   
            recording_ids = [i.decode('utf-8') for i in dataset[f'{label}_labels']]
            data = {k:v for k, v in tqdm(zip(recording_ids, dataset[f'{label}_files']), total=len(recording_ids))}

            data_value = np.zeros(shape, dtype=np.float32)
            print('data_value.shape:',data_value.shape)
            for i in tqdm(range(len(data))):
                keyname = str(i)+'_'+label
                data_value[i] = data[keyname]

        return data_value
    
    # 前半，後半データを使ってそれぞれモデルを学習し，お互いの正解ラベルに異常ラベルを追加する  
    def co_fit(self, num_rep, batch_size, epochs):
        '''
        num_rep:ラベル追加の繰り返し回数
        batch_size:訓練する際のバッチサイズ
        
        '''
        each_data_trained_model = self.each_data_trained_model
        NormalSave = Sound_load_and_save(self.sr, self.duration, self.melparams, way=True)
        
        for count in range(num_rep):
            each_data_updated_df_group =[]
            for data_path_group in self.each_data_path_group:
                label_df = [] # 正解ラベルを格納したリスト
                for data_path in data_path_group:          
                    label_data_name = data_path.split('/')[-1].replace('.wav', '.csv') 
                    label_data_file = self.each_data_update_dir+label_data_name
                    label_data_df=pd.read_csv(label_data_file, skiprows=1, usecols=[1], header=None)    # fileをデータフレームに出力
                    label_df.append(label_data_df)
                each_data_updated_df_group.append(label_df)
            
            
            #################################################################################################################         
            # 前半，後半データの正常音をつかってそれぞれのモデルを学習させる
            for i, (datas, datasLabel_df, npy_output, hdf_output) in enumerate(zip(self.each_data_path_group, each_data_updated_df_group, self.each_data_npy_output, self.each_data_hdf_output)):
                
                #npyファイルに変換
                print('npy_output:', npy_output)
                if(os.path.isdir(npy_output) == True):
                    print(f'{npy_output}が既に存在するため削除します')
                    shutil.rmtree(npy_output)
                    print(f'{npy_output}削除完了')
                os.makedirs(npy_output, exist_ok=False)

                _ = joblib.Parallel(n_jobs=-1)(joblib.delayed(NormalSave.load_and_save)(i,j,k,l) for i,j,k,l in tqdm(zip(datas, [npy_output]*len(datas), [self.transform]*len(datas), datasLabel_df), total=len(datas)))
                print(f'{npy_output}にnpyファイルを保存しました．')
                gc.collect() #回収可能なオブジェクトを削除
                
                # スペクトログラムを3チャンネルにして，HDF5ファイルに変換
                print('hdf_output:', hdf_output)
                if os.path.exists(hdf_output):
                    print(f'{hdf_output}が既に存在するため削除します')
                    os.remove(hdf_output)
                    print(f'{hdf_output}削除完了')
                SaveAsHdf5 = save_as_hdf5(hdf_output, npy_output) 
                SaveAsHdf5()
                print(f'{hdf_output}にhdfファイルを保存しました')
                gc.collect() #回収可能なオブジェクトを削除
                
                # データロード用のジェネレータオブジェクトの生成
                with h5py.File(hdf_output, mode='r') as dataset:
                    train_data_size = len(dataset['train_files'])
                    val_data_size = len(dataset['val_files'])
                    train_data_shape = dataset['train_files'].shape

                train_key_name = 'train_files'
                val_key_name = 'val_files'
                train_datagen = genSequence(hdf_output, train_data_size, train_key_name, batch_size)
                val_datagen = genSequence(hdf_output, val_data_size, val_key_name, batch_size)                    

                # オートエンコーダの定義
                autoencoder = self.make_autoencoder(name=os.path.basename(hdf_output)) # 前半，後半のモデルを区別できるように名前を指定
                autoencoder.compile(optimizer='adam', loss='mse')
                autoencoder.summary()

                # オートエンコーダの学習
                history = autoencoder.fit(
                    train_datagen, 
                    epochs=epochs,
                    validation_data=val_datagen,
                    callbacks=[TensorBoard(log_dir='/tmp/autoencoder'), ClearMemory()]
                )        
                
                # オートエンコーダの保存
                trained_model_save_path = each_data_trained_model[i]+str(count)+'_autoencoder.hd5'
                print('trained_model_save_path:', trained_model_save_path)
                if os.path.exists(trained_model_save_path):
                    print(f'{trained_model_save_path}が既に存在するため削除します')
                    os.remove(trained_model_save_path)
                    print(f'{trained_model_save_path}削除完了')
                autoencoder.save(trained_model_save_path, save_format='h5') 
                print(f'[{trained_model_save_path}]にモデルを保存しました．')

                pd.DataFrame({'loss':history.history['loss'], 'val_loss':history.history['val_loss']}).plot()    
                
                # いったんモデルを削除してリセット
                tf.keras.backend.clear_session() # 計算グラフを破棄する
                del autoencoder                       # 変数を削除する   
                gc.collect() #回収可能なオブジェクトを削除
                
            
            ##########################################################################
            #前半データと後半データをロードしてラベルを追加
            for i, (datas, datasLabel_array) in enumerate(zip(self.each_data_path_group, self.each_data_original_array_group)):
                # オートエンコーダのロード
                print(f'datas:{datas}')
                print('(i-1)*(-1)が0:前半データの正常音で学習させたモデル')
                print('(i-1)*(-1)が1:後半データの正常音で学習させたモデル')
                print(f'(i-1)*(-1):{(i-1)*(-1)}')
                trained_model_save_path = each_data_trained_model[(i-1)*(-1)]+str(count)+'_autoencoder.hd5'
                autoencoder = models.load_model(trained_model_save_path) # 正常音で学習させたモデルのロード
                autoencoder.summary()

                # 閾値の決定   
                find_threshold = FindThreshold(self.threshold_data_path, '正常音', autoencoder, 256, 4.0)# 正常音に対する異常度の最大値を閾値とする   
                _, abnorm_thr = find_threshold() 
                print(f'閾値：{abnorm_thr}')
                
                tf.keras.backend.clear_session() # 計算グラフを破棄する
                gc.collect() #回収可能なオブジェクトを削除
                
                for data, label_array in zip(datas, datasLabel_array):
                    label = os.path.splitext(os.path.basename(data))[0]
                    print('label:', label)
                    predict_data_value = self.load_dataset(label, self.eachday_hdf)# 時系列順にデータをndarrayに格納する
                    addlabel = AddLabel(label_array, self.duration, abnorm_thr, way=True) #異常度が閾値よりも大きいところに異常ラベルを追加する

                    #ラベルの追加
                    added_label = addlabel.predict_batch_add_label(predict_data_value, 256, autoencoder)
                    
                    if count<num_rep-1:
                        save_path = self.each_data_update_dir+ label+ '.csv'
                    else:
                        save_path = self.each_data_final_dir + label+ '.csv' # 繰り返しの最後の時は最終結果のフォルダに保存する
                    
                    if os.path.exists(save_path):
                        print(f'{save_path}が既に存在するため削除します')
                        os.remove(save_path)
                        print(f'{save_path}削除完了')
                    added_label_df = pd.DataFrame(added_label) # 最終的なラベルのデータフレーム
            #         display(added_label_df)
                    added_label_df.to_csv(save_path)

                # いったんモデルを削除してリセット
                tf.keras.backend.clear_session() # 計算グラフを破棄する
                del autoencoder                       # 変数を削除する
                gc.collect() #回収可能なオブジェクトを削除
        
#     @profile                
    def all_fit(self, batch_size, epochs):
        NormalSave = Sound_load_and_save(self.sr, self.duration, self.melparams, way=True)
        
        all_train_data = self.each_data_path_group[0]+self.each_data_path_group[1] # 全訓練データのパスのリスト

        # 訓練データの更新した正解ラベルのDFをリストにまとめる
        data_updated_df =[]
        for data_path in all_train_data:     
            label_data_name = data_path.split('/')[-1].replace('.wav', '.csv') 
            label_data_file = self.each_data_final_dir +label_data_name
            label_data_df=pd.read_csv(label_data_file, skiprows=1, usecols=[1], header=None)    # fileをデータフレームに出力
            print(label_data_file)
            display(label_data_df)
            data_updated_df.append(label_data_df)
        
        #npyファイルに変換
        npy_output = './data_output/npy_files/全訓練データ/'
        print('npy_output:', npy_output)
        if(os.path.isdir(npy_output) == True):
            print(f'{npy_output}が既に存在するため削除します')
            shutil.rmtree(npy_output)
            print(f'{npy_output}削除完了')
        os.makedirs(npy_output, exist_ok=False)

        _ = joblib.Parallel(n_jobs=-1)(joblib.delayed(NormalSave.load_and_save)(i,j,k,l) for i,j,k,l in tqdm(zip(all_train_data, [npy_output]*len(all_train_data), [self.transform]*len(all_train_data), data_updated_df), total=len(all_train_data)))
        print(f'{npy_output}にnpyファイルを保存しました．')
        gc.collect() #回収可能なオブジェクトを削除        

        
        # スペクトログラムを3チャンネルにして，HDF5ファイルに変換
        hdf_output=self.alltraindata_hdf_output
        print('hdf_output:', hdf_output)
        if os.path.exists(hdf_output):
            print(f'{hdf_output}が既に存在するため削除します')
            os.remove(hdf_output)
            print(f'{hdf_output}削除完了')
        SaveAsHdf5 = save_as_hdf5(hdf_output, npy_output) 
        self.train_mean, self.train_std = SaveAsHdf5()# テストデータを標準化する際に使用する訓練データの平均値と標準偏差を受け取る
        print(f'訓練データ平均値：{self.train_mean}')
        print(f'訓練データ標準偏差：{self.train_std}')
        print(f'{hdf_output}にhdfファイルを保存しました')
        gc.collect() #回収可能なオブジェクトを削除       
        
        
        # データロード用のジェネレータオブジェクトの生成
        with h5py.File(hdf_output, mode='r') as dataset:
            train_data_size = len(dataset['train_files'])
            val_data_size = len(dataset['val_files'])
            train_data_shape = dataset['train_files'].shape

        train_key_name = 'train_files'
        val_key_name = 'val_files'
        train_datagen = genSequence(hdf_output, train_data_size, train_key_name, batch_size)
        val_datagen = genSequence(hdf_output, val_data_size, val_key_name, batch_size)                    

        # オートエンコーダの定義
        autoencoder = self.make_autoencoder(name=os.path.basename(hdf_output)) # 前半，後半のモデルを区別できるようにnameを指定
        autoencoder.compile(optimizer='adam', loss='mse')
        autoencoder.summary()

        # オートエンコーダの学習
        history = autoencoder.fit(
            train_datagen, 
            epochs=epochs,
            validation_data=val_datagen,
            callbacks=[TensorBoard(log_dir='/tmp/autoencoder')]
        )        

        # オートエンコーダの保存
        trained_model_save_path = './data_output/trainedmodel/alltraindata/' 
        os.makedirs(trained_model_save_path, exist_ok=True)
        
        import datetime
        dt_now = datetime.datetime.now() # 誤って上書きしないように現在時刻をファイル名に入れる
        trained_model_save_path = trained_model_save_path+dt_now.strftime('%Y%m%d%H%M%S')+'_sslautoencoder.hd5'
        print('trained_model_save_path:', trained_model_save_path)
        if os.path.exists(trained_model_save_path):
            print(f'{trained_model_save_path}が既に存在するため削除します')
            os.remove(trained_model_save_path)
            print(f'{trained_model_save_path}削除完了')
        autoencoder.save(trained_model_save_path, save_format='h5') 
        print(f'[{trained_model_save_path}]にモデルを保存しました．')

        pd.DataFrame({'loss':history.history['loss'], 'val_loss':history.history['val_loss']}).plot()    

        # いったんモデルを削除してリセット
        tf.keras.backend.clear_session() # 計算グラフを破棄する
        del autoencoder                       # 変数を削除する   
        gc.collect() #回収可能なオブジェクトを削除        


    def predict(self,predict_data_path, model_path, xbox, xdiff):
        '''
        predict_data_path：予測に使用するwavデータのパス
        model_path :学習済みモデルのパス     
        xbox:四分位範囲を何倍するか
        xdiff:差分画像を何倍するか
        戻り値：異常度の配列(ndarray)と学習済みモデルをつかって正常音，異常音データから算出した閾値のタプル
        '''        
        autoencoder = models.load_model(model_path)
        autoencoder.summary()

        # 閾値の決定
        find_threshold = FindThreshold(self.threshold_data_path, '正常音', autoencoder, 256, xbox)  
        _, norm_max = find_threshold() 
        abnorm_thr = norm_max
        print(f'閾値：{abnorm_thr}')
        
        tf.keras.backend.clear_session() # 計算グラフを破棄する
        gc.collect() #回収可能なオブジェクトを削除     
        
        print(f'self.train_mean:{self.train_mean}')
        print(f'self.train_std:{self.train_std}')        
        if (self.train_mean==None)or(self.train_std==None):
            with h5py.File(self.alltraindata_hdf_output, mode='r') as dataset: 
                self.train_mean = dataset['mean'][...]  
                self.train_std = dataset['std'][...]
                print(f'with文内,mean:{self.train_mean}')
                print(f'with文内,std:{self.train_std}')                  
        # テストデータをメルスペクトログラムに変換しhdf形式で保存
        testdata_hdf_path = './data_output/hdf5/testdata_'+os.path.splitext(os.path.basename(predict_data_path))[0]+'.hdf5'   
        test_save_as_hdf5=TestSaveAsHdf5(predict_data_path, self.duration, self.melparams)
        testdata_hdf_path = test_save_as_hdf5(self.train_mean, self.train_std)
        gc.collect() #回収可能なオブジェクトを削除   
        
        # 予測データを時系列順のndarrayに格納する
        label = os.path.splitext(os.path.basename(predict_data_path))[0]
        print('label:', label)
        predict_data_value = self.load_dataset(label, testdata_hdf_path )        

        # 異常度の計算
        predict_batch_size=256
        decoded_imgs = np.empty(predict_data_value.shape, dtype=np.float32)  
        buried_sound = np.zeros(len(predict_data_value)) # 異常音が埋もれているかどうかの判断につかう

        BATCH_INDICES = np.arange(start=0, stop=len(predict_data_value), step=predict_batch_size)  
        BATCH_INDICES = np.append(BATCH_INDICES, len(predict_data_value)) 

        print(len(BATCH_INDICES) - 1)
        for index in tqdm(np.arange(len(BATCH_INDICES) - 1)):
            batch_start = BATCH_INDICES[index]  
            batch_end = BATCH_INDICES[index + 1] 
            decoded_imgs[batch_start:batch_end] = autoencoder.predict_on_batch(predict_data_value[batch_start:batch_end]) 
            
            # 差分画像の最大値と復元画像の平均値の比を使ってBG音に埋もれているかどうかを判断.
            diff = (predict_data_value[batch_start:batch_end]-decoded_imgs[batch_start:batch_end])     
            diff_max = diff.mean(axis=3).max(axis=1).max(axis=1) 
            decoded_imgs_mean = decoded_imgs[batch_start:batch_end].mean(axis=3).mean(axis=1).mean(axis=1)        
            buried_sound[batch_start:batch_end] = diff_max/decoded_imgs_mean
            
        anomary_scores = np.zeros(len(decoded_imgs))
        for i in range(len(anomary_scores)):
            anomary_scores[i] = np.mean((decoded_imgs[i]-predict_data_value[i])**2)    
                    
        buried_sound  = np.where(buried_sound>=xdiff, 1, 0) # １なら埋もれていない，0なら埋もれている   
        notburied_anomary_scores = np.where(buried_sound==1, anomary_scores, 0)      

        tf.keras.backend.clear_session() # 計算グラフを破棄する
        del autoencoder                       # 変数を削除する   
        gc.collect() #回収可能なオブジェクトを削除              
            
        return anomary_scores, notburied_anomary_scores, abnorm_thr

    
    
    
    
    
    
    
    