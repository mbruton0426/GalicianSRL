#!/bin/bash

# trains all models introduced in GalicianSRL Project; results for each model are output to outputs/[model_name].txt

# install dependencies
pip install transformers==4.27.4 
pip install datasets==2.11.0 
pip install evaluate==0.4.0 
pip install seqeval==1.2.2

# don't forget to save final model to specific file, then recall as local model to train the gal_spa models in the .py files

# train Spanish, then dependant Galician models
python scripts/spa_mBERT.py > outputs/spa_mBERT-gal_sp_mBERT.txt
python scripts/spa_en_mBERT.py > outputs/spa_en_mBERT-gal_ensp_mBERT.txt
python scripts/spa_pt_mBERT.py > outputs/spa_pt_mBERT-gal_ptsp_mBERT.txt
python scripts/spa_enpt_mBERT.py > outputs/spa_enpt_mBERT-gal_enptsp_mBERT.txt
python scripts/spa_XLM-R.py > outputs/spa_XLM-R-gal_sp_XLM-R.txt
python scripts/spa_en_XLM-R.py > outputs/spa_en_XLM-R-gal_ensp_XLM-R.txt
python scripts/spa_pt_XLM-R.py > outputs/spa_pt_XLM-R-gal_ptsp_XLM-R.txt
python scripts/spa_enpt_XLM-R.py > outputs/spa_enpt_XLM-R-gal_enptsp_XLM-R.txt

# train remaining Galician models
python scripts/gal_mBERT.py > outputs/gal_mBERT.txt
python scripts/gal_en_mBERT.py > outputs/gal_en_mBERT.txt
python scripts/gal_pt_mBERT.py > outputs/gal_pt_mBERT.txt
python scripts/gal_enpt_mBERT.py > outputs/gal_enpt_mBERT.txt
python scripts/gal_XLM-R.py > outputs/gal_XLM-R.txt
python scripts/gal_en_XLM-R.py > outputs/gal_en_XLM-R.txt
python scripts/gal_pt_XLM-R.py > outputs/gal_pt_XLM-R.txt
python scripts/gal_enpt_XLM-R.py > outputs/gal_enpt_XLM-R.txt
