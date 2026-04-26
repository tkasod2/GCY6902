%load_ext autoreload
%autoreload 2
from run_classification import *
from make_sample_df import *
# from run_inference import *
%matplotlib inline

plt.rc('font', family='Malgun Gothic')
df = make_sample_df(n_features=10)
df = df.rename(columns={"F010":"DEFAULT"})
df['DEFAULT'] = np.log1p(abs(df['DEFAULT']))
df['DEFAULT'] = np.random.permutation(df['DEFAULT'].values)

%%
outdir = "./output/" + datetime.now().strftime('%y%m%d%H%M%S')+"/"
os.mkdir(outdir)
var_list = [x for x in df.columns if x.startswith('F')]+['DEFAULT']

config = {'d_model':64,
    'hidden_dim': 4,#128
    'lstm_hidden': 4,#128
    'n_heads': 4,
    'dropout': 0.2,

    'past_vars': len(var_list),
    'known_vars':  1,   # 예: time index, month, quarter 등
    'static_vars':  1,
    'output_mode': "multiclass",  # "regression" | "binary" | "multiclass"
    'n_classes': 3,
}

fit_and_out(df,'DEFAULT', outdir,config,var_list,epochs=1,patience=1,threshold=0.05,output_mode='multiclass')