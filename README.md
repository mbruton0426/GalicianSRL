# BERTie Bott's Every Flavor Labels: A Tasty Guide to Developing a Semantic Role Labeling Model for Galician
This repository was published as part of my Master's thesis in Language Technology. It created the first publically available Galician SRL dataset, a Spanish SRL dataset utilizing a verbal indexing method introduced as part of the work, and 24 SRL models. Fully trained models are available on [HuggingFace](https://huggingface.co/mbruton) alongside more specific information on each model.

## Reproducing Dataset Development
To reproduce the development of the Galician Dataset, both the [CTG UD annotated corpus](https://github.com/UniversalDependencies/UD_Galician-CTG) and the [TreeGal UD annotated corpus](https://github.com/UniversalDependencies/UD_Galician-TreeGal) CoNLL files must be downloaded. Save files as, "data/gl_ctg-ud-dev.conllu", "data/gl_ctg-ud-test.conllu", "data/gl_ctg-ud-train.conllu", "data/gl_treegal-ud-test.conllu", and "data/gl_treegal-ud-train.conllu". Utilizing the `create_dataset_galician.py` script from this repository, you can then run the command:
```bash
python create_dataset_galician.py
```
This combines all sentences from both corpora, removes duplicates, indexes verbs, and assigns arguments. A final Dataset will be saved locally to "data/final_gal_ds.hf". This may differ from the [GalicianSRL](https://huggingface.co/datasets/mbruton/galician_srl) Dataset published on HuggingFace as some manual validation was performed on the published Dataset.

To reproduce the Spanish Dataset, the 2009 CoNLL Spanish data must be downloaded. Save files as, "data/CoNLL2009-ST-Spanish-development.txt", "data/CoNLL2009-ST-Spanish-train.txt", and "data/CoNLL2009-ST-evaluation-Spanish.txt". Utilizing the `create_dataset_spanish.py` script from this repository, you can then run the command:
```bash
python create_dataset_spanish.py
```
This utilizes only information contained in the 2009 CoNLL Spanish data and indexes verbs. Three final Datasets will be saved locally to "data/spa_srl_ds_dev.hf", "data/spa_srl_ds_train.hf", and "data/spa_srl_ds_test.hf". To preserve the dev/train/test data splits, these Datasets were then manually combined to created the [SpanishSRL](https://huggingface.co/datasets/mbruton/spanish_srl) Dataset published on HuggingFace, but they are able to be used independently as well.

## Reproducing Model Development
To reproduce all models at once, all 16 scripts located in the `scripts` folder within this repository must be downloaded and stored locally in "scripts/". A folder "outputs/" should be created as well, as this is where results for each model will be output as "outputs/[model_name].txt". The `train.sh` script can then be utilized by running the command:
```bash
sh train.sh
```
If only specific models need reproduced, dependencies must be manually installed using the command:
```bash
pip install transformers==4.27.4 
pip install datasets==2.11.0 
pip install evaluate==0.4.0 
pip install seqeval==1.2.2
```
The proper script can then be located within the `scripts` folder and ran using the command:
```bash
python [model_name].py
```
Python scripts are named using the following convention: [finetuning_language]_[pretraining_language]_[base_model]. finetuning_language can appear as "gal" for Galician or "spa" for Spanish. pretraining_language can appear as "en" for English, "pt" for Portuguese, "sp" for Spanish, or some combination of the three. base_model can appear as "mBERT" or "XLM-R" each refering to their respective base model.
