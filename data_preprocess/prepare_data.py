import os
import subprocess
import textgrid as textgrid_utils
import argparse
import pathlib

def create_tsv(file_name, path, files):
    with open(file_name, 'w') as f:
        f.write(path + '\n')
    for i, file in enumerate(files):
        with open(file_name, 'a+') as f:
            f.write(f'{file}\t{i}\n')


def convert_wav_2_16hz(path, verbose):
    """
    convert all wav files in path to 16khz 16bit wav files
    path: path to the folder containing wav files
    replace: if True, replace the original wav files with the converted ones, else, create new files with _cnv suffix
    """
    files = [f for f in os.listdir(path) if f.endswith(".wav") and not f.startswith(".")]
    assert len(files) > 0, f"no wav files found in {path}"
    outs = []
    for file in files:
        input_f = os.path.join(path, file)
        out_f = input_f.replace('.wav', '_cnv.wav')
        if verbose:
            print(f"running sox {input_f} -r 16000 -b 16 {out_f}")
        pout = subprocess.run(f"sox {input_f} -r 16000 -b 16 {out_f}".split())
        if pout.returncode != 0:
            outs.append(file)

    print(f"Done converting waves.")
    if len(outs) > 0:
        print(f"{len(outs)} files had errors when converting. Check convert_log.txt for  the file names")
        with open('convert_log.txt', 'a+') as f:
            f.write("\n".join(outs))


def prepare_data_dir(path, tsv_name, resample_waves, tag_tier_name, word_tier_name, split_by_sentence, verbose, suffix="", create_tsv=True):
    """
    prepare data within a directory folder. convert all wav files to 16khz 16bit and then splits waves to sub waves by sentence.
    path: path to the folder containing wav files
    dataset_name: name of the dataset
    tsv_prefix: prefix for the tsv file name
    resample_waves: if True, convert all wav files to 16khz 16bit
    tag_tier_name: name of the tier containing the sentence
    word_tier_name: name of the tier containing the words tags
    """
    sentences = []
    tg_err = []

    if resample_waves:
        convert_wav_2_16hz(path, verbose)

    if split_by_sentence:
        textgrid_files = [os.path.join(path, f) for f in
                        os.listdir(path) if f.endswith('.TextGrid')]
        textgrid_files.sort()
        assert len(textgrid_files) > 0, f"no textgrid files found in {path}"
        if verbose:
            print("Found the following textgrids:", textgrid_files)
        for textgrid_file in textgrid_files:
            if textgrid_file.startswith('.'):
                continue

            if verbose:
                print("Processing textgrid file:", textgrid_file)
            tg_path = textgrid_file
            spkr_tg = textgrid_utils.TextGrid().fromFile(tg_path)

            for t_name, w_name in [(tag_tier_name, word_tier_name), ('sentence', 'sentence - phones'), ('sentence - words', 'sentence - phones')]:
                tag_tier = spkr_tg.getFirst(t_name)
                word_tier = spkr_tg.getFirst(w_name)
                if tag_tier is not None and word_tier is not None:
                    break
            if tag_tier is None or word_tier is None:
                tg_err.append(tg_path)
                continue

            for int_id, interval in enumerate(tag_tier):
                if interval.mark != "":
                    start_time, end_time, txt = interval.minTime, interval.maxTime, interval.mark
                    word_alignment = [(start_time, end_time, txt)]
                    # To avoid some tagging errors, get correct sentence region by using the word tier.
                    for wrd_id, word_interval in enumerate(word_tier):
                        if word_interval.overlaps(interval) and word_interval.mark not in ["", 'sp', 'sil']:
                            word_alignment.append((word_interval.minTime, word_interval.maxTime, word_interval.mark))
                    start_time = word_alignment[1][0]
                    end_time = word_alignment[-1][1]
                    duration = end_time - start_time
                    suffix = "_cnv.wav" if resample_waves else ".wav"
                    input_file = tg_path.replace(".TextGrid", suffix)
                    out_file = tg_path.replace(".TextGrid", f"_sent{int_id}.wav")
                    if verbose:
                        print(f"speaker {out_file}: sentence {int_id} start {start_time} duration {duration}")
                        print("running ", f"sox {input_file} {out_file} trim {start_time} {duration}")
                    with open(out_file.replace('.wav', '.TextGrid'), 'w') as f:
                        for line in word_alignment:
                            f.write(f"{line[0]},{line[1]},{line[2]}\n")

                    subprocess.run(f"sox {input_file} {out_file} trim {start_time} {duration}".split())
                    sentences.append(os.path.basename(out_file))
        if len(tg_err) > 0:
            print(f"found {len(tg_err)} files with no text tier:", tg_err)    
    else:
        if suffix == "":
            suffix = "_cnv.wav" if resample_waves else ".wav"
        files = [f for f in os.listdir(path) if f.endswith(suffix) and not f.startswith(".")]
        for file in files:
            sentences.append(file)
    
    assert len(sentences) > 0, f"no wav files found in {path}"
    if create_tsv:
        create_tsv_in_tsvs(path, tsv_name, sentences)
    print("Preprocessing done.")
    return sentences

def create_tsv_in_tsvs(path, tsv_name, sentences):
    curr_dir_path = pathlib.Path(__file__).parent.parent.resolve()
    tsv_path = os.path.join(curr_dir_path, f"tsvs/{tsv_name}.tsv")
    create_tsv(tsv_path, path, sentences)

def prepare_all_star(path, resample_waves=True, tag_tier_name='utt', word_tier_name='speaker - word', split_by_sentence=True, verbose=False):
    """
    prepare ALLSTAR (HT1,HT2,LPP,DHR) data. convert all wav files to 16khz 16bit wav files and then splits waves to sentence waves.
    """
    reading_tasks = {'cmn_HT1':[], 'cmn_HT2':[], 'cmn_LPP':[], 'cmn_DHR':[], 'spn_HT1':[], 'spn_HT2':[], 'spn_LPP':[], 'spn_DHR':[]}
    for reading_task_dir in os.listdir(path):
        if reading_task_dir.startswith(".") or not os.path.isdir(os.path.join(path, reading_task_dir)):
            continue

        reading_task = reading_task_dir.split('_')[-1]
        if 'SHS_SPA' in reading_task_dir:
            word_tier_name = tag_tier_name

        print(f"Preparing ALLSTAR {reading_task_dir} data")
        sentences = prepare_data_dir(os.path.join(path, reading_task_dir), "ALLSTAR_"+reading_task_dir, resample_waves,
                         tag_tier_name, word_tier_name, split_by_sentence, verbose, create_tsv=False)
        for i in range(len(sentences)):
            sentences[i] = os.path.join(reading_task_dir, os.path.basename(sentences[i]))
        if 'ENG_ENG' in reading_task_dir:
            reading_tasks['cmn_'+reading_task].extend(sentences)
            reading_tasks['spn_'+reading_task].extend(sentences)
        else:
            l2dir = 'cmn_' if 'CMN' in reading_task_dir else 'spn_'
            reading_tasks[l2dir+reading_task].extend(sentences)

    for reading_task, sentences in reading_tasks.items():
        if len(sentences) > 0:
            create_tsv_in_tsvs(path, "ALLSTAR_"+reading_task, sentences)
        else:
            print(f"no data for {reading_task}")

argparser = argparse.ArgumentParser()
argparser.add_argument('--path', type=str, default='', help='path to the data directory')
argparser.add_argument('--tsv_name', type=str, default='', help='name of the tsv file for custom dataset')
argparser.add_argument('--resample_waves', action='store_true', help='convert waves to 16khz 16bit')
argparser.add_argument('--tag_tier_name', type=str, default='utt', help='name of the tier containing the sentence')
argparser.add_argument('--word_tier_name', type=str, default='Speaker - word', help='name of the tier containing the words tags')
argparser.add_argument('--prepare_allstar', action='store_true', help='whether to prepare all star data instead of a single directory with custom dataset')
argparser.add_argument('--split_by_sentence', action='store_true', help='whether to split wave files to smaller chuncks by sentence using textgrid files')
argparser.add_argument('--verbose', action='store_true', help='')
argparser.add_argument('--tsv_suffix', type=str, default="", help='')


if __name__ == '__main__':
    args = argparser.parse_args()
    if args.prepare_allstar:
        prepare_all_star(args.path, args.resample_waves, args.tag_tier_name, args.word_tier_name, args.split_by_sentence, args.verbose)
    else:
        prepare_data_dir(args.path, args.tsv_name, args.resample_waves, args.tag_tier_name, args.word_tier_name, args.split_by_sentence, args.verbose, args.tsv_suffix)
