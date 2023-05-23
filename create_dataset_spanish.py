# imports
from datasets import Dataset, ClassLabel, Sequence, Features, Value, load_from_disk
from collections import defaultdict
import pandas as pd


# functions
def import_conll_09(file_path):
    sent_dict = defaultdict(dict)
    sent_count = 0
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        word_dict = defaultdict(dict)
        sent = []
        verbs = []
        
        for i, line in enumerate(lines):
            line = line.split()
            word_idx = None
            
            try:
                word_idx = line[0]
            except:
                continue
            
            if word_idx:
                if word_idx == '1' and i != 0: # new sent
                    sent_dict[sent_count]['sent'] = sent
                    sent_dict[sent_count]['verbs'] = verbs
                    sent_dict[sent_count]['sent_data'] = word_dict
                    sent = []
                    verbs = []
                    word_dict = defaultdict(dict)
                    sent_count += 1
                    
                
                word = line[1]
                pos_list = line[4:6]
                head = line[8]
                sentiment = line[13]
                arg_list = line[14:]
                
                for argument in arg_list:
                    if len(argument) > 1:
                        arg = argument
                        break
                    else:
                        arg = 'O'
                        
                if 'v' in pos_list and sentiment != '_':
                    verbs.append(word_idx)
                
                word_dict[word_idx]['word'] = word
                word_dict[word_idx]['head'] = head
                word_dict[word_idx]['arg'] = arg
                
                sent.append(word)
    f.close()        
    
    X = []
    y = []

    for sent_idx in sent_dict:
        nsentence = sent_dict[sent_idx]['sent']
        nsent_verbs = sent_dict[sent_idx]['verbs']
        nsent_data = sent_dict[sent_idx]['sent_data']

        X.append(sent_dict[sent_idx]['sent'])

        mini_y = []
        
        for word_idx in nsent_data:
            if word_idx in nsent_verbs:
                verb_idx = nsent_verbs.index(word_idx)
                narg = 'r' + str(verb_idx) + ':root'
                mini_y.append(narg)
            else:
                ntok = nsent_data[word_idx]['word']
                nhead = nsent_data[word_idx]['head']
                aarg = nsent_data[word_idx]['arg'].translate({ord('-'):'|'})
                
                if nhead in nsent_verbs and aarg != 'O' and 'null' not in aarg:
                    verb_idx = nsent_verbs.index(nhead)
                    narg = 'r' + str(verb_idx) + ':' + aarg
                    mini_y.append(narg)
                else:
                    mini_y.append('O')
        y.append(mini_y) 

    for i, w in enumerate(X):
        if len(w) != len(y[i]):
            print(w)
            print(y[i])
            print(len(w), len(y[i]))
            print(' ')
    
    final_dict = {
        'tokens' : X,
        'tags' : y,
    }
        
    return final_dict           


if __name__ == "__main__":
    # convert Spanish files to matching format for verbal indexing
    try:
        spa_dict_dev = import_conll_09('data/CoNLL2009-ST-Spanish-development.txt')
        spa_dict_train = import_conll_09('data/CoNLL2009-ST-Spanish-train.txt')
        spa_dict_test = import_conll_09('data/CoNLL2009-ST-evaluation-Spanish.txt')
    except:
        print('Error! Cannot find CoNLL 2009 Spanish files. Please double check you have downloaded all files and that they are stored correctly.')
        print('Files should be stored as: "data/CoNLL2009-ST-Spanish-development.txt", "data/CoNLL2009-ST-Spanish-train.txt", and "data/CoNLL2009-ST-evaluation-Spanish.txt".')


    # assign sentence IDs
    dev_sent_ids = [i for i in range(len(spa_dict_dev['tokens']))]
    spa_dict_dev['ids'] = dev_sent_ids

    test_sent_ids = [i for i in range(len(spa_dict_train['tokens']))]
    spa_dict_train['ids'] = test_sent_ids

    train_sent_ids = [i for i in range(len(spa_dict_test['tokens']))]
    spa_dict_test['ids'] = train_sent_ids


    # ensure data combined correctly
    assert len(spa_dict_dev['tags']) == len(spa_dict_dev['tokens'])
    assert len(spa_dict_train['tags']) == len(spa_dict_train['tokens'])
    assert len(spa_dict_test['tags']) == len(spa_dict_test['tokens'])


    # define labels
    spa_dev_labels = sorted(set(label for labels in spa_dict_dev['tags'] for label in labels))
    spa_train_labels = sorted(set(label for labels in spa_dict_train['tags'] for label in labels))
    spa_test_labels = sorted(set(label for labels in spa_dict_test['tags'] for label in labels))
    all_labels = sorted(set(spa_dev_labels + spa_train_labels + spa_test_labels))


    # convert to Dataset object and convert labels to ClassLabel
    spa_dev_ds = Dataset.from_dict(spa_dict_dev)
    spa_train_ds = Dataset.from_dict(spa_dict_train)
    spa_test_ds = Dataset.from_dict(spa_dict_test)
    
    final_spa_dev_ds = spa_dev_ds.cast_column('tags', Sequence(ClassLabel(names=all_labels)))
    final_spa_test_ds = spa_test_ds.cast_column('tags', Sequence(ClassLabel(names=all_labels)))
    final_spa_train_ds = spa_train_ds.cast_column('tags', Sequence(ClassLabel(names=all_labels)))


    # save Dataset objects locally
    final_spa_dev_ds.save_to_disk('data/spa_srl_ds_dev.hf')
    final_spa_test_ds.save_to_disk('data/spa_srl_ds_test.hf')
    final_spa_train_ds.save_to_disk('data/spa_srl_ds_train.hf')

    # can be recalled individually or manually combined into a single Dataset
    
    # individually
    # final_spa_dev_ds = load_from_disk('data/spa_srl_ds_dev.hf')
    
    # manually combine
    # create folder 'data/final_gal_ds.hf'
    # rename and move 'data/spa_srl_ds_dev.hf' to 'data/final_spa_ds.hf/dev'
    # rename and move 'data/spa_srl_ds_train.hf' to 'data/final_spa_ds.hf/train'
    # rename and move 'data/spa_srl_ds_test.hf' to 'data/final_spa_ds.hf/test'
    # place 'dataset_dict.json' into 'data/final_spa_ds.hf' folder
    # final_spa_ds = load_from_disk('data/final_spa_ds.hf')