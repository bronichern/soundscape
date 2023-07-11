import os
import numpy as np
import pandas as pd
from distances import parallel_dtw, mut_normalize_sequences, librosa_dtw
import argparse
from sklearn.manifold import TSNE
import re

DTW = 0
DTWPAR = 1

CMN = 0
KR = 1
SPN = 2

TSNE_COL=['tsne-2d-one', 'tsne-2d-two', 'tsne-2d-thr']
dist_name = ['DTW', 'DTWPAR']


def load_speakers_metadata(feats_file, tsv_name):
    '''
    Loads time indexes for each speaker and their names
    '''
    speakers_len = open(feats_file.replace('.npy', '.len'), 'r').read().split('\n')
    speakers_len = np.array([int(f) for f in speakers_len if f != ''])

    wav_paths = re.split('\t\d\n|\n', open("tsvs/"+tsv_name, 'r').read())
    names = np.array([f.split(".")[0] for f in wav_paths[1:] if f != ""])
    return speakers_len, names


def create_df(data_group, feats, speaker_len, names):
    '''
    Creates a dataframe with the features and the speaker information. Each row is a time step of a speaker.
    data_group: CMN, KR, SPN
    speaker_len: list of the number of time steps for each speaker
    names: list of speaker names
    '''

    cols = [f"val {i}" for i in range(feats.shape[1])]
    df = pd.DataFrame(feats, columns=cols)
    df['index'] = df.index
    # match speaker to rows
    time_index = {i: speaker_len[i] for i in range(len(speaker_len))}
    com_time_index = {i: sum(speaker_len[:i]) for i in range(len(speaker_len))}
    df_speaker_count = pd.Series(time_index)
    df_speaker_count = df_speaker_count.reindex(df_speaker_count.index.repeat(
        df_speaker_count.to_numpy())).rename_axis('speaker_id').reset_index()
    df['speaker_id'] = df_speaker_count['speaker_id']
    df['speaker_len'] = df['speaker_id'].apply(lambda row: speaker_len[row])
    df['com_sum'] = df['speaker_id'].apply(lambda i: com_time_index[i])
    df['speaker'] = df['speaker_id'].apply(lambda i: names[i])

    assert len(df_speaker_count) == len(df)

    if data_group == CMN:
        df['path'] = df['speaker'].str.split('/').str[0]
        df['speaker'] = df['speaker'].str.split('/').str[1]
    elif data_group == SPN:
        df['path'] = df['speaker'].str.split('/').str[0]
        df['speaker'] = df['speaker'].str.split('/').str[-1]

    assert len(df.loc[df['speaker'] == -1]) == 0
    df_subset = df.copy()
    data_group = df_subset[cols].values
    return data_group, df_subset, cols


def tsne(data_subset, init='random', early_exaggeration=12.0, lr=200.0, n_comp=2, perplexity=40, iters=300):
    tsne = TSNE(n_components=n_comp, verbose=1, perplexity=perplexity, n_iter=iters,
                init=init, early_exaggeration=early_exaggeration,
                learning_rate=lr, random_state=12312)
    tsne_results = tsne.fit_transform(data_subset)
    return tsne_results


def data_group_name(data_group):
    return 'CMN' if data_group == CMN else 'KR' if data_group == KR else 'SPN'
        
def build_kr_df(args):
    feat_path_data = args.feat_path
    tsv_name_data = args.tsv_name

    feats_file_path = os.path.join(
        feat_path_data, f"kr_layer{args.layer}", tsv_name_data.replace('.tsv', '_0_1.npy'))

    feats = np.load(feats_file_path)
    speakers_len, names = load_speakers_metadata(feats_file_path, tsv_name_data)
    if not args.run_all_data:
        snts = list(set([f.split("_")[2] for f in names]))
        snts.sort()
        splt_len = int(len(snts)/9)
        sub_portion = (args.portion-1)*splt_len
        curr_snts = snts[sub_portion: sub_portion +
                            splt_len] if args.portion < 9 else snts[sub_portion:]
        idxs = [i for i, f in enumerate(names) if f.split("_")[2] in curr_snts]

        feat_idxs = []
        for i, id in enumerate(idxs):
            # treat each speaker individ. and load his idxs - up tohim is start id and speakers_len[id] is length
            feat_idxs.extend(range(sum(speakers_len[:id]), sum(speakers_len[:id]) + speakers_len[id]))
        speakers_len = speakers_len[idxs]
        names = names[idxs]
        feats = feats[feat_idxs]

    data_subset, df_subset, cols = create_df(args.data_group, feats, speakers_len, names)
    df_subset['idx'] = df_subset.index
    df_subset['speaker_main'] = df_subset['speaker'].apply(lambda i: "_".join(i.split(
        "_")[1:2]))
    df_subset['speaker_par'] = df_subset['speaker'].apply(lambda i: "_".join(i.split(
        "_")[2:3]))
    df_subset['sentLNG'] = df_subset['speaker_par'].apply(lambda i: i[:2])
    df_subset['sent_cross_unq'] = df_subset['sentLNG']+ df_subset['speaker_par']
    return data_subset, df_subset, speakers_len, cols


def build_cmn_df(args):
    files_to_load = []
    feat_path_data = args.feat_path
    tsv_name_data = args.tsv_name

    feat_file_task = 'ht' if args.reading_task == '' or 'ht' in args.reading_task.lower(
    ) else args.reading_task.lower()
    with open('tsvs/'+tsv_name_data, 'r') as f:
        data_path = f.read().split('\n')

    if args.data_group == CMN:
        all_paths = [data_path[0]]+list(set([os.path.join(data_path[0], f.split('/')[
                                        0]) for f in data_path[1:] if args.reading_task in f]))
        assert len(all_paths) == 4, all_paths
        print(all_paths)
        feats_file_data = os.path.join(
            feat_path_data, f"{data_group_name(args.data_group)}_{feat_file_task}_layer{args.layer}",
            tsv_name_data.replace('.tsv', '_0_1.npy'))

    else:
        all_paths = [data_path[0]]+list(set([os.path.join(data_path[0], f.split('/')[0], f.split(
            '/')[1]) for f in data_path[1:] if args.reading_task in f and 'ENG_ENG' not in f]))
        all_paths += list(set([os.path.join(data_path[0], f.split('/')[0])
                            for f in data_path[1:] if args.reading_task in f and 'ENG_ENG' in f]))
    
        assert len(all_paths) == 4

    for ap in all_paths[1:]:
        files_to_load += [os.path.join(ap, f)
                            for f in os.listdir(ap) if '.wav' in f and 'sent' in f]
    
    feats = np.load(feats_file_data)
    speakers_len, names = load_speakers_metadata(feats_file_data, tsv_name_data)
    
    if not args.run_all_data:
        idxs = [i for i, f in enumerate(names) if os.path.join(
            all_paths[0], f+'.wav') in files_to_load and f'sent{args.portion}' in f]
        name2idx = {f.split('/')[1]: i for i, f in enumerate(names) if (os.path.join(
            all_paths[0], f+'.wav') in files_to_load) and f'sent{args.portion}' in f}
    
        feat_idxs = []
        for i, id in enumerate(idxs):
            # treat each speaker individ. and load his idxs - up tohim is start id and speakers_len[id] is length
            feat_idxs.extend(range(sum(speakers_len[:id]), sum(speakers_len[:id]) + speakers_len[id]))
        speakers_len = speakers_len[idxs]
        names = names[idxs]
        feats = feats[feat_idxs]

    data_subset, df_subset, cols = create_df(
        args.data_group, feats, speakers_len, names)
    df_subset['idx'] = df_subset.index
    
    df_subset.path = all_paths[0]+df_subset.path
    df_subset['speaker_main'] = df_subset['speaker'].apply(
        lambda i: "_".join(i.split("_")[1:4]))
    df_subset['speaker_par'] = df_subset['speaker'].apply(
        lambda i: "_".join(i.split("_")[3:6]))
    df_subset['speaker_LNG'] = df_subset['speaker'].apply(
        lambda i: "_".join(i.split("_")[3:4]))
    df_subset['rng'] = df_subset['speaker'].apply(lambda r: name2idx[r])
    df_subset['sentid'] = df_subset['speaker'].apply(lambda i:i.split("_")[-1].split("sent")[1])
    df_subset['sent_cross_unq'] =  df_subset['speaker'].apply(lambda i:"_".join(i.split("_")[4:5]))+df_subset.sentid
    return data_subset, df_subset, speakers_len, cols


def measure_dist(filename, use_tsne, sub, speakers, cols, data_group, verbose=False):
    normalize = True
    normalize_by_len = True
    norm_func = mut_normalize_sequences
    mode = 'FULL_DIM' if not use_tsne else 'TSNE'
    dist_df = pd.DataFrame(
        columns=["speaker1", "speaker2", "group", "part", "sent"])

    if use_tsne:
        distance_func, func_name = librosa_dtw, 'SoundscapeDistance'
    else:
        distance_func, func_name = parallel_dtw, 'SoundscapeDistance'

    if data_group == KR:
        sent_lng = speakers[0].split("_")[2]
    elif data_group in [CMN, SPN]:
        sent_lng = speakers[0].split("_")[5]

    for sp_id, spkr in enumerate(speakers[:-1]):
        for spkr2 in speakers[sp_id+1:]:
            if data_group == KR:
                spkr1_sent_lng = spkr.split("_")[2]
                spkr2_sent_lng = spkr2.split("_")[2]

                spkr_l1 = spkr.split("_")[1][:1]
                spkr2_l1 = spkr2.split("_")[1][:1]
            elif data_group in [CMN, SPN]:
                spkr_l1 = spkr.split("_")[3][:2]
                spkr2_l1 = spkr2.split("_")[3][:2]
                spkr1_sent_lng = spkr.split("_")[4][:2]
                spkr2_sent_lng = spkr2.split("_")[4][:2]
                spkr1_sent = spkr.split("_")[-1]
                spkr2_sent = spkr2.split("_")[-1]
                spkr1_task = spkr.split("_")[5]
                spkr2_task = spkr2.split("_")[5]

                if spkr1_sent != spkr2_sent or spkr1_task != spkr2_task:
                    continue
            group = 'BT' if spkr_l1[0] != spkr2_l1[0] else spkr_l1[0]
            if spkr1_sent_lng != spkr2_sent_lng:
                continue
            if verbose:
                print(f"Measuring distance for spkr {spkr} and {spkr2}")

            sent_id = spkr1_sent if data_group in [CMN, SPN] else ""
            dist_row = {"speaker1": spkr, "speaker2": spkr2,
                        "group": group, "part": sent_lng, "sent": sent_id}

            if use_tsne:
                tsne_spkr = sub[(sub.speaker == spkr)][TSNE_COL].to_numpy()
                tsne_spkr2 = sub[(sub.speaker == spkr2)][TSNE_COL].to_numpy()
            else:
                tsne_spkr = sub[(sub.speaker == spkr)][TSNE_COL].to_numpy()
                tsne_spkr2 = sub[(sub.speaker == spkr2)][TSNE_COL].to_numpy()

            tsne_spkr, tsne_spkr2 = norm_func(
                tsne_spkr, tsne_spkr2, normalize)
            dist_val = distance_func(tsne_spkr, tsne_spkr2)

            if type(dist_val) == tuple and len(dist_val) > 0:
                dist_val = dist_val[0]
            if normalize_by_len:
                dist_val = dist_val/(len(tsne_spkr)+len(tsne_spkr2))
            dist_row[f"{func_name}"] = dist_val

            dist_df = dist_df.append(dist_row, ignore_index=True)

    out_name = f"{filename}_{mode}.csv"
    dist_df.to_csv(out_name)


def fill_tsne(df_subset,ids, tsne_results):
    df_subset.loc[ids,('tsne-2d-one')] = tsne_results[:,0]
    df_subset.loc[ids,('tsne-2d-two')] = tsne_results[:,1]
    if tsne_results.shape[1] == 3:
        df_subset.loc[ids,('tsne-2d-thr')] = tsne_results[:,2]
    return df_subset


def main(args):
    use_tsne = args.project
    if args.data_group == KR:
        data_subset, df, speakers_len, cols = build_kr_df(args)
    else:
        data_subset, df, speakers_len, cols = build_cmn_df(args)

    df['new_id'] = list(range(df.shape[0]))
    metric = dist_name[args.dist_group]

    if args.data_group in [CMN,SPN]:
        task = args.reading_task.lower()
        if not os.path.exists(os.path.join(args.output_path, f"{task}")):
            os.mkdir(os.path.join(args.output_path, f"{task}"))
        if use_tsne:
            filename = os.path.join(
                args.output_path, f"{task}", f"{args.data}_{args.file_pref}_{metric}_tsnedim{args.tsne_dim}_layer{args.layer}_portion{args.portion}_"
                + df.iloc[0].speaker_par.split("_")[-1])
        else:
            filename = os.path.join(
                args.output_path, f"{task}", f"{args.data}_{args.file_pref}_{metric}_FULLDIM_layer{args.layer}_portion{args.portion}_"
                + df.iloc[0].speaker_par.split("_")[-1])
    else:
        filename = os.path.join(
            args.output_path, f"{args.data}_{args.file_pref}_{metric}_tsnedim{args.tsne_dim}_layer{args.layer}_portion{args.portion}_"
            + df.iloc[0].speaker_par.split("_")[-1])

    print(
        f"Running distnace evaluation for layer {args.layer} with t-SNE projection {use_tsne}")
    print("File will be saved to ", filename)
    ids = np.array(df.index)       
    tsne_res = tsne(data_subset, init='pca', early_exaggeration=2.0, lr=100.0,n_comp=args.tsne_dim, perplexity = 40, iters = 300)
    df = fill_tsne(df,ids,tsne_res)
    measure_dist(filename, use_tsne, df, df.speaker.unique(), cols, args.data_group, args.verbose)


parser = argparse.ArgumentParser(description='Parser for distance evaluation',)
parser.add_argument('--layer', type=int, required=True)
parser.add_argument('--project', action='store_true')
parser.add_argument('--data', type=str, required=True,
                    help='Type of data group: cmn, kr')
parser.add_argument('--portion', type=int, default=0)
parser.add_argument('--file_pref', type=str, default="", help="Prefix for output file name")
parser.add_argument('--tsv_name', type=str, required=True)
parser.add_argument('--output_path', type=str, required=True,
                    help="Path where distance outputs will be stored (csv format)")
parser.add_argument('--feat_path', type=str, required=True,
                    help="Path to where Hubert's .npy feature files are stored")
parser.add_argument('--reading_task', type=str, default=None)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--run_all_data', action='store_true')
parser.add_argument('--tsne_dim', type=int, default=3)
args = parser.parse_args()

if __name__ == '__main__':
    if args.project:
        print(f'evaluating distance with {args.tsne_dim} dimensions')
        args.dist_group = DTW
    else:
        print('evaluating distance with all dimensions')
        args.dist_group = DTWPAR
    if args.data.lower() == 'cmn':
        args.data_group = CMN
    elif args.data.lower() == 'kr':
        args.data_group = KR
    elif args.data.lower() == 'spn':
        args.data_group = SPN
    else:
        raise Exception(
            f"Invalid data passed {args.data} | Possible values are 'cmn', 'spn' or 'kr'")
    if args.data_group != KR and args.reading_task is None:
        raise Exception(
            f"For ALLSTAR dataset you must specify reading_task and portion")
    main(args)
