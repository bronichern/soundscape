# soundscape

## 1. Installation
- Follow the installation instructions of fairseq: https://pypi.org/project/fairseq/  
- Run:
```
pip install -r requirements.txt
```

## 2. Using the trajectory demo
To render the notebook plots, use nbviewer. For instance, for the Korean dataset notebook, use:  
https://nbviewer.org/github/bronichern/soundscape/blob/main/kr_trajectories.ipynb
  
To change the plots, you need hubert's features.  To change the korean notebook, you can download the files [Here](https://drive.google.com/drive/folders/1ZkDLYDqN9BWv_5frwNLi_lRvexw0f0a0?usp=share_link)
To generate other features, see the instructions in *section 3*.  
  
Instead of the notebook you can also run a script that will open the plots directly in a dedicated view:  
- Assuming the files above were downloaded to: `/home/some_path` (change that with the actual location where you downloaded the files). 
- Now `/home/some_path` has 2 files: `kr_en_full_0_1.npy` and `kr_en_full_0_1.len`  
- In this repo, there is a yaml config file: `demo/kr_trajectories.yaml`.   Edit the value of `feat_path` in the yaml to have the path `/home/some_path`.  Then, dit the value of tsv_path, to the tsvs directory in this repor.  
- Run the following - change `path_to_config` to point to the location of the yaml file above:  
```
python demo/kr_trajectories.py --path_to_config
```

## 3. Data preparation
The following script can resample the data to 16khz, split it by sentence (if textgrids are available) and create a tsv file in tsvs/.
```
python data_preprocess/prepare_data.py --path data_path --resample_waves --split_by_sentence --tsv_name name_of_tsv_file
```
Remove ```--resample_waves``` if you don't want to resample the files and remove ```--split_by_sentence``` if the files are relatively short(~1 second up to ~18 seconds) and there is no need for splitting by sentence/word.  
```name_of_tsv_file``` - should be without '.tsv' at the end.
```data_path``` - directory with wave files.  
- If splitting is relevant, textgrids should be within the same directory and have the same name as the corresponding wave file.

### Korean dataset
```
python data_preprocess/prepare_data.py --path kr_path --resample_waves --tsv_name kr
```

### ALLSTAR dataset
```
python data_preprocess/prepare_data.py --path allstar_path --resample_waves --split_by_sentence --prepare_allstar
```
Note that the script will create a tsv for each reading task in the dataset.

## 4. Create a directory for Hubert's feature file:
Throughout the next sections, there is a distinction between the ALLSTAR dataset, which has different reading tasks, and the Korean dataset which doesn't.  
### Korean dataset
Within the directory you choose to store Hubert's feature file, create a directory named kr_layer```layer-number``` (i.e kr_layer1)  
```layer-number``` - Hubert layer we want to have as a feature extractor  
### ALLSTAR Mandarin/Spanish dataset
Within the directory you choose to store Hubert's feature file, create a directory named  ```dataset_name```\_```reading_task```\_layer```layer-number``` (i.e cmn_lpp_layer1)  
```dataset_name``` - 'cmn' or 'spn' for Mandarin or Spanish respectively.  
```reading_task``` - "HT1"/"HT2"/"LPP"/"DHR".  
```layer-number``` - Hubert layer we want to have as a feature extractor.  

## Running DTW+MFCC baseline

## Running Soundscape distance
```
python run.py --reading_task task_name --data data_subset_name --layer hubert_layer --portion sentences_portion --verbose --feat_path hubert_feature_path --tsv_name tsv_name --project --output_path output_csv_dir
```
 ```--data```: First, to know which dataset we process, ```--data``` argument is required. ```--data cmn``` or ```--data spn``` are for the ALLSTAR dataset, with L1 English speakers and L2 english speakers with L1 Mandarin or L1 Spanish, respectively; ```--data kr``` specifies to use the Korean dataset with L1 English speakers and L2 English speakers with Korean L1. The different datasets have different processing when creating their DataFrame.  
```--portion```: Next, since there was a lot of data to process, the script processes subsets of the data. Meaning since we divided the waves by sentences, ```--portion``` accepts integers between 1-9 that specifies to process sentences with an id that begins with ```sentences_portion```. This parameter is only relevant for the ALLSTAR dataset (i.e with arguments ```--data cmn``` or ```--data spn```). 
- If you wish to run using all the data instead of specifing ```--portion```, use ```--run_all_data```.

```--reading_task``` is relevant for the ALLSTAR dataset where task_name can be one of "HT1","HT2","LPP","DHR".  
```--project``` argument is specified to project using TSNE. Remove this argument if you don't want to project the data. If you wish to use 2 dimension for TSNE instead of 3 (default), specify ```--tsne_dim 2```.  
At the end of processing a CSV file will be created in the directory ```output_csv_dir``` specified in the argument ```--output_path output_csv_dir```.  


The script assumes that the required Hubert's feature file is saved in - see the next section for instruciton on how to create Hubert feature file: ```hubert_feature_path```/```data_subset_name```\_```task_name```\_```hubert_layer```/  

For instance, for running ALLSTAR-HT1 dataset with L1 English speakers and L2 Manadarin speakers, with a feature file for layer 12 saved in hubert_feature_path/cmn_ht1_layer12, and with a tsv file created in tsvs/allstar_ht1.tsv run the following command:  
```
python run.py --reading_task "HT1" --data "cmn" --layer 12 --portion 1 --verbose --feat_path hubert_feature_path/ --tsv_name allstar_ht1.tsv --project --output_path output_csv_dir
```

For running the dataset with L1 English speakers and L2 Korean speakers, with a feature file for layer 12 saved in hubert_feature_path/kr_ht1_layer12, and with a tsv file created in tsvs/kr.tsv run the following command:  
```
python run.py --data "kr" --layer 12 --portion 1 --verbose --feat_path hubert_feature_path/ --tsv_name kr.tsv --project --output_path output_csv_dir
```

## Generating feature file
The following section uses fairseq's repo, which was installed inside this repo when you run the installation in *section 1*.   
Download Hubert base model - [Here](https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt).  
Run the following to create the feature file:  
```
cd fairseq
python examples/hubert/simple_kmeans/dump_hubert_feature.py tsvs tsv_name path_of_hubert_model layer_number 1 0 output_path_for_features
```
```layer_number``` - Hubert layer we want to have as a feature extractor  
```tsv_name``` - the tsv file created in *section 3*)  
 ```output_path_for_features``` is the path for saving the feature file. The one created in *section 4* (i.e. cmn_ht1_layer12)
