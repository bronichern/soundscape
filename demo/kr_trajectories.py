import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import re
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import yaml
import argparse

def load_speakers_metadata(feats_file, tsv_name):
    speaker_len = open(feats_file.replace('.npy','.len'),'r').read().split('\n')
    speaker_len = [int(f) for f in speaker_len if f != '']

    wav_paths = re.split('\t\d\n|\n',open(tsv_name, 'r').read())
    names = [f.split(".")[0] for f in wav_paths[1:] if f!=""]
    
    return speaker_len, names 

def time_txt(time, time_frame=5):
    if time % time_frame == 0:
        return f"{round(time*0.02,2)}"
    return ""

def create_df(feats, speaker_len, names):
    cols = [f"val {i}" for i in range(feats.shape[1])]
    df = pd.DataFrame(feats,columns = cols)
    df['index'] = df.index
    time_index = {i:speaker_len[i] for i in range(len(speaker_len))}
    com_time_index = {i:sum(speaker_len[:i]) for i in range(len(speaker_len))}
    df_speaker_count = pd.Series(time_index)
    df_speaker_count = df_speaker_count.reindex(df_speaker_count.index.repeat(df_speaker_count.to_numpy())).rename_axis('speaker_id').reset_index()
    df['speaker_id'] = df_speaker_count['speaker_id']
    df['speaker_len']  = df['speaker_id'].apply(lambda row: speaker_len[row])
    df['com_sum'] = df['speaker_id'].apply(lambda i: com_time_index[i])
    df['speaker'] = df['speaker_id'].apply(lambda i: names[i])
    assert len(df.loc[df['speaker']==-1]) == 0
    assert len(df_speaker_count) == len(df)
    df_subset = df.copy()
    data_subset = df_subset[cols].values
    return data_subset,df_subset, cols 

def tsne(data_subset, init='random', early_exaggeration=12.0, lr = 200.0, n_comp = 2, perplexity = 40, iters = 300,random_state=None):
    tsne = TSNE(n_components=n_comp, verbose=1, perplexity=perplexity, n_iter=300,init=init, early_exaggeration=early_exaggeration,
               learning_rate = lr,random_state=random_state)
    tsne_results = tsne.fit_transform(data_subset)
    return tsne_results

def fill_tsne(df_subset, tsne_results,tsne_1,tsne_2,tsne_3):
    print(tsne_results[:,0].shape)
    df_subset[tsne_1] = tsne_results[:,0]
    df_subset[tsne_2] = tsne_results[:,1]
    if tsne_results.shape[1] == 3:
        df_subset[tsne_3] = tsne_results[:,2]
    return df_subset


def plot_tsne(df_subset, tsne_1, tsne_2, tsne_3):
    import plotly.offline as pyo
    import plotly.graph_objs as go
    
    pyo.init_notebook_mode()
    fig = px.scatter_3d(df_subset, x=tsne_1, y=tsne_2, z=tsne_3,
              color='speaker')
    fig.update_traces(mode='lines+markers+text')
    pyo.iplot(fig, filename='jupyter-styled_bar')


class CFG:
    def __init__(self, **entries): 
        self.__dict__.update(entries)

argparser = argparse.ArgumentParser()
argparser.add_argument('--config_path', type=str, default='kr_trajectories_config.yaml')
arg_dict = argparser.parse_args()

if __name__ == "__main__":
    with open(arg_dict.config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    cfg = CFG(**cfg_dict)
    if cfg.feat_path == "":
        print("Please specify the path to the features")
        exit()
    feat_path, layer, tsv_name = cfg.feat_path, cfg.layer, cfg.tsv_name
    seed = cfg.seed
    time_frame = cfg.time_frame
    sentence_id = cfg.sentence_id

    feats_file = os.path.join(feat_path,layer,tsv_name.replace('.tsv','_0_1.npy'))
    feats = np.load(feats_file)
    speaker_len, names = load_speakers_metadata(feats_file, "tsvs/"+tsv_name)
    print(f"running feats {feats_file}")

    data_subset,df_subset, cols = create_df(feats, speaker_len, names)
    df_subset['idx'] = df_subset.index
    df_subset['speaker_main'] = df_subset['speaker'].apply(lambda i: i.split("_")[1])
    df_subset['speaker_par'] = df_subset['speaker'].apply(lambda i: i.split("_")[2])
    df_subset['time'] = df_subset['idx']-df_subset['com_sum']
    df_subset['time_txt'] = df_subset[['time','speaker_len']].apply(lambda row: time_txt(row['time'],time_frame),axis=1)
    df_subset['L1']=df_subset.speaker_main.apply(lambda r: "EN" if r[0]  == 'E' else "K")


    df_subset_orig = df_subset.copy()
    data_subset_orig = data_subset.copy()
    tsne_1 = 'tsne-3d-one'
    tsne_2 = 'tsne-3d-two'
    tsne_3 = 'tsne-3d-thr'


    df_subset = df_subset_orig.copy()
    data_subset= data_subset_orig.copy()
    df_subset= df_subset.loc[df_subset.speaker_par == sentence_id]
    data_subset = data_subset[df_subset.loc[df_subset.speaker_par == sentence_id].index]
    tsne_results = tsne(data_subset, init='pca', early_exaggeration=2.0, lr=100.0,n_comp=3, perplexity = 40, iters = 1000, random_state=seed)
    df_subset = fill_tsne(df_subset, tsne_results, tsne_1, tsne_2, tsne_3)
    
    S1 = cfg.speaker1
    S2 = cfg.speaker2
    dcp = df_subset.loc[((df_subset['speaker_main'].isin([S1,S2]))&(df_subset['speaker_par']==sentence_id))].copy().rename(columns={tsne_1: "x",tsne_2:'y',tsne_3:'z'})
    dcp1 = dcp.loc[(dcp['speaker_main']==S1)].copy()
    dcp2 = dcp.loc[(dcp['speaker_main']==S2)].copy()
    dcp1['clr'] = np.linspace(0, 1, dcp.loc[(dcp['speaker_main']==S1)].shape[0])
    dcp2['clr'] = np.linspace(1, 0, dcp.loc[(dcp['speaker_main']==S2)].shape[0])

    # S1
    fig = px.scatter_3d(dcp1, x='x', y='y', z='z',
                        color='clr',symbol='speaker_main',
                        text='time_txt',
                        labels={'x':'t-SNE-dim1','y':'t-SNE-dim2','z':'t-SNE-dim3'})
    fig.update_traces(marker_symbol='diamond',marker_coloraxis=None,marker_colorscale='burg', mode='lines+markers+text', line_color='red')
    fig.for_each_trace(lambda t: t.update(textfont_color='darkred'))

    # S2
    fig2 = px.scatter_3d(dcp2, x='x', y='y', z='z',
                        color='clr',symbol='speaker_main',
                        text='time_txt',
                        labels={'x':'t-SNE-dim1','y':'t-SNE-dim2','z':'t-SNE-dim3'})
    fig2.update_traces(marker_coloraxis=None,marker_colorscale='ice', mode='lines+markers+text', line_color='blue')
    fig2.for_each_trace(lambda t: t.update(textfont_color='blue'))

    fig3 = go.Figure(data=fig.data + fig2.data)
    fig3.update_layout(scene = dict(
                        xaxis_title='dimension 1 (t-SNE)',
                        yaxis_title='dimension 2 (t-SNE)',
                        zaxis_title='dimension 3 (t-SNE)',
    ),
                        
                        margin=dict(r=20, b=10, l=10, t=10),
                        legend_title="Speaker",)
    fig3.show()