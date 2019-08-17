import warnings
import numpy as np
import pandas as pd



from scipy import stats
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")



def extract_feature(index,signal,arr):
    #get signal data
    signals_no_val=pd.Series(signal['acoustic_data']) #it will be use later
    signals = pd.Series(signal['acoustic_data'].values)
    #get fourier transform of function
    arr.loc[index, 'mean'] = signals.mean()
    arr.loc[index, 'std'] = signals.std()
    arr.loc[index, 'max'] = signals.max()
    arr.loc[index, 'min'] = signals.min()
    #in pieces of normal distrubition
    arr.loc[index, 'q01'] = np.quantile(signals,0.01)
    arr.loc[index, 'q05'] = np.quantile(signals,0.05)
    arr.loc[index, 'q95'] = np.quantile(signals,0.95)
    arr.loc[index, 'q99'] = np.quantile(signals,0.99)
    #abs
    arr.loc[index, 'abs_max'] = np.abs(signals).max()
    arr.loc[index, 'abs_median'] = np.median(np.abs(signals))
    arr.loc[index, 'ave10'] = stats.trim_mean(signals, 0.1)
    arr.loc[index, 'max_to_min_diff'] = signals.max() - np.abs(signals.min())
    arr.loc[index, 'max_first_50000'] = signals[:50000].max()
    arr.loc[index, 'max_last_50000'] = signals[-50000:].max()
    arr.loc[index, 'max_first_10000'] = signals[:10000].max()
    arr.loc[index, 'max_last_10000'] = signals[-10000:].max()
    arr.loc[index, 'min_first_50000'] = signals[:50000].min()
    arr.loc[index, 'min_last_50000'] = signals[-50000:].min()
    arr.loc[index, 'min_first_10000'] = signals[:10000].min()
    arr.loc[index, 'min_last_10000'] = signals[-10000:].min()

    window_size=[50,100]
    
    for i in window_size:
        x_rolling_std = signals_no_val.rolling(i).std().dropna().values
        x_rolling_mean = signals_no_val.rolling(i).mean().dropna().values
        arr.loc[index, 'ave_roll_std_' + str(i)] = x_rolling_std.mean()
        arr.loc[index, 'std_roll_std_' + str(i)] = x_rolling_std.std()
        arr.loc[index, 'max_roll_std_' + str(i)] = x_rolling_std.max()
        arr.loc[index, 'min_roll_std_' + str(i)] = x_rolling_std.min()
        arr.loc[index, 'q01_roll_std_' + str(i)] = np.quantile(x_rolling_std,0.01)
        arr.loc[index, 'q05_roll_std_' + str(i)] = np.quantile(x_rolling_std,0.05)
        arr.loc[index, 'q95_roll_std_' + str(i)] = np.quantile(x_rolling_std,0.95)
        arr.loc[index, 'q99_roll_std_' + str(i)] = np.quantile(x_rolling_std,0.99)
        arr.loc[index, 'ave_roll_mean_' + str(i)] = x_rolling_mean.mean()
        arr.loc[index, 'std_roll_mean_' + str(i)] = x_rolling_mean.std()
        arr.loc[index, 'max_roll_mean_' + str(i)] = x_rolling_mean.max()
        arr.loc[index, 'min_roll_mean_' + str(i)] = x_rolling_mean.min()
        arr.loc[index, 'q01_roll_mean_' + str(i)] = np.quantile(x_rolling_mean,0.01)
        arr.loc[index, 'q05_roll_mean_' + str(i)] = np.quantile(x_rolling_mean,0.05)
        arr.loc[index, 'q95_roll_mean_' + str(i)] = np.quantile(x_rolling_mean,0.95)
        arr.loc[index, 'q99_roll_mean_' + str(i)] = np.quantile(x_rolling_mean,0.99)
    extract_fft_feature(index,signal,arr)
    
def extract_fft_feature(index,signal,arr):
    #arr ->>>>> train data ---->X
    #get signal data
    
    signals_no_val=pd.Series(signal['acoustic_data']) #it will be use later
    signals = pd.Series(signal['acoustic_data'].values)
    
    values = np.fft.fft(signals)
    realValues = np.real(values) #İgnored imaginer part
    signals=realValues
    #same process
    arr.loc[index, 'fft_Rmean'] = signals.mean()
    arr.loc[index, 'fft_Rstd'] = signals.std()
    arr.loc[index, 'fft_Rmax'] = signals.max()
    arr.loc[index, 'fft_Rmin'] = signals.min()
    #in pieces of normal distrubition
    arr.loc[index, 'fft_q01'] = np.quantile(signals,0.01)
    arr.loc[index, 'fft_q05'] = np.quantile(signals,0.05)
    arr.loc[index, 'fft_q95'] = np.quantile(signals,0.95)
    arr.loc[index, 'fft_q99'] = np.quantile(signals,0.99)
    #abs 
    arr.loc[index, 'fft_abs_max'] = np.abs(signals).max()
    arr.loc[index, 'fft_abs_median'] = np.median(np.abs(signals))
    arr.loc[index, 'fft_ave10'] = stats.trim_mean(signals, 0.1)
    arr.loc[index, 'fft_max_to_min_diff'] = signals.max() - np.abs(signals.min())
    arr.loc[index, 'fft_max_first_50000'] = signals[:50000].max()
    arr.loc[index, 'fft_max_last_50000'] = signals[-50000:].max()
    arr.loc[index, 'fft_max_first_10000'] = signals[:10000].max()
    arr.loc[index, 'fft_max_last_10000'] = signals[-10000:].max()
    arr.loc[index, 'fft_min_first_50000'] = signals[:50000].min()
    arr.loc[index, 'fft_min_last_50000'] = signals[-50000:].min()
    arr.loc[index, 'fft_min_first_10000'] = signals[:10000].min()
    arr.loc[index, 'fft_min_last_10000'] = signals[-10000:].min()

    window_size=[50,100]
    
    for i in window_size:
        x_rolling_std = signals_no_val.rolling(i).std().dropna().values
        x_rolling_mean = signals_no_val.rolling(i).mean().dropna().values
        arr.loc[index, 'fft_ave_roll_std_' + str(i)] = x_rolling_std.mean()
        arr.loc[index, 'fft_std_roll_std_' + str(i)] = x_rolling_std.std()
        arr.loc[index, 'fft_max_roll_std_' + str(i)] = x_rolling_std.max()
        arr.loc[index, 'fft_min_roll_std_' + str(i)] = x_rolling_std.min()
        arr.loc[index, 'fft_q01_roll_std_' + str(i)] = np.quantile(x_rolling_std,0.01)
        arr.loc[index, 'fft_q05_roll_std_' + str(i)] = np.quantile(x_rolling_std,0.05)
        arr.loc[index, 'fft_q95_roll_std_' + str(i)] = np.quantile(x_rolling_std,0.95)
        arr.loc[index, 'fft_q99_roll_std_' + str(i)] = np.quantile(x_rolling_std,0.99)
        arr.loc[index, 'fft_ave_roll_mean_' + str(i)] = x_rolling_mean.mean()
        arr.loc[index, 'fft_std_roll_mean_' + str(i)] = x_rolling_mean.std()
        arr.loc[index, 'fft_max_roll_mean_' + str(i)] = x_rolling_mean.max()
        arr.loc[index, 'fft_min_roll_mean_' + str(i)] = x_rolling_mean.min()
        arr.loc[index, 'fft_q01_roll_mean_' + str(i)] = np.quantile(x_rolling_mean,0.01)
        arr.loc[index, 'fft_q05_roll_mean_' + str(i)] = np.quantile(x_rolling_mean,0.05)
        arr.loc[index, 'fft_q95_roll_mean_' + str(i)] = np.quantile(x_rolling_mean,0.95)
        arr.loc[index, 'fft_q99_roll_mean_' + str(i)] = np.quantile(x_rolling_mean,0.99)

samples = 150_000

train = pd.read_csv("./data/train.csv",
                    dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})

train_signal_df = train['acoustic_data']
train_failure_df = train['time_to_failure']
segments = int(np.floor(train.shape[0]/samples))

train_X = pd.DataFrame(index=range(segments), dtype=np.float64)
train_y = pd.DataFrame(index=range(segments), dtype=np.float64)


for df_id in tqdm(range(segments)):
    signal = train.iloc[df_id*samples:df_id*samples+samples]
    extract_feature(df_id,signal,train_X)
    for k in (range(samples//100,samples,samples//100)):
        train_y.loc[df_id, 'time_to_failure_'+str(k)] = signal['time_to_failure'].values[k]
    train_y.loc[df_id, 'time_to_failure_'+str(samples)] = signal['time_to_failure'].values[-1]

scaler = StandardScaler()
scaler.fit(train_X)
scaled_train_X = pd.DataFrame(scaler.transform(train_X), columns=train_X.columns)
scaled_train_X

scaled_train_X.to_csv("train_X.csv", sep=',', encoding='utf-8')
train_y.to_csv("train_y.csv", sep=',', encoding='utf-8')

submission = pd.read_csv('./sample_submission.csv', index_col='seg_id')

X_test = pd.DataFrame(columns=scaled_train_X.columns, dtype=np.float64, index=submission.index)

for seg_id in tqdm(X_test.index):
    seg = pd.read_csv('./test/' + seg_id + '.csv')
    extract_feature(seg_id, seg, X_test)

X_test_scaled = scaler.transform(X_test)
np.savetxt("X_test_scaled.csv", X_test_scaled, delimiter=",")





