# imports
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('propbank')
nltk.download('treebank')
from nltk.corpus import propbank
from nltk.corpus import wordnet as wn
from datasets import Dataset, ClassLabel, Sequence, Features, Value, load_from_disk
from collections import defaultdict
import pandas as pd


# functions
def ctg_to_conll_dict(file, conll_dict=None, gal_to_propbank_dict=None):
    if conll_dict==None and gal_to_propbank_dict==None:
        conll_dict = defaultdict(dict)
        gal_to_propbank_dict = defaultdict(dict)
        
    elif conll_dict==None or gal_to_propbank_dict==None:
        return 'Error! Must provide either both or neither conll_dict and gal_to_propbank_dict.'
    else:
        conll_dict = conll_dict
        gal_to_propbank_dict = gal_to_propbank_dict
            
    with open(file, 'r') as f:
        lines = f.readlines()
        sent_id = None

        for line in lines:
            if 'sent_id' in line:
                if sent_id != None:
                    conll_dict[sent_id]['sent'] = sent
                    conll_dict[sent_id]['sent_data'] = sent_data_dict

                line = line.strip().split(' ')
                line = line[-1].split('-')
                sent_id = int(line[-1])
                
                sent_data_dict = defaultdict(dict)
                
                sent = None
                word_index = None
                word_form = None
                lemma = None
                upos = None
                xpos = None
                feats = None
                head = None
                deprel = None
                deps = None
                misc = None

            elif '#' in line:
                line = line.strip().split('=')
                sent = line[-1]
            else:
                line = line.split('\t')

                if len(line) < 2:
                    continue
                else:
                    word_index = line[0]
                    word_form = line[1]
                    lemma = line[2]
                    upos = line[3]
                    xpos = line[4]
                    feats = line[5]
                    head = line[6]
                    deprel = line[7]
                    deps = line[8]
                    misc = line[9].strip()

                    sent_data_dict[word_index]['word_form'] = word_form
                    sent_data_dict[word_index]['lemma'] = lemma
                    sent_data_dict[word_index]['upos'] = upos
                    sent_data_dict[word_index]['xpos'] = xpos
                    sent_data_dict[word_index]['feats'] = feats
                    sent_data_dict[word_index]['head'] = head
                    sent_data_dict[word_index]['deprel'] = deprel
                    sent_data_dict[word_index]['deps'] = deps
                    sent_data_dict[word_index]['misc'] = misc
                    sent_data_dict[word_index]['role'] = '_'
                                        
                    if upos == 'VERB':
                        lemma_synsets = wn.synsets(lemma, lang='glg')
                        rolesets = []
                        arg_sets = []
                        
                        for synset in lemma_synsets:
                            split_synset = str(synset).split("'")
                            split_split = split_synset[1].split('.')
                            syn = split_split[0] + '.' + split_split[-1] #'garner.01'
                            
                            try:
                                roleset = propbank.roleset(syn)
                                rolesets.append(syn)
                                args = dict()
                                
                                for role in roleset.findall('roles/role'):
                                    arg_num = 'A' + role.attrib['n']
                                    args[arg_num] = role.attrib['descr']
                                    
                                arg_sets.append(args)
                            
                            except:
                                continue # no match
                            
                        gal_to_propbank_dict[lemma]['rolesets'] = rolesets
                        gal_to_propbank_dict[lemma]['arg_sets'] = arg_sets
                        
                        if len(rolesets) == 0:
                            sent_data_dict[word_index]['role'] = 'undefined.01'
                            sent_data_dict[word_index]['args'] = {'A0' : 'describer',
                                                                  'A1' : 'thing defined',
                                                                  'A2' : 'attribute'}
                        elif len(rolesets) == 1:
                            sent_data_dict[word_index]['role'] = rolesets[0]
                            sent_data_dict[word_index]['args'] = arg_sets[0]
                        
                        else:
                            sent_data_dict[word_index]['role'] = 'see gal_to_propbank_dict'
                            sent_data_dict[word_index]['args'] = {'A0' : 'describer',
                                                                  'A1' : 'thing defined',
                                                                  'A2' : 'attribute'}
                
                        
    f.close()
    return conll_dict, gal_to_propbank_dict


def ctg_conll_add_args(conll_dictionary):

    for sent_id in conll_dictionary.keys(): # each sent
        sent = conll_dictionary[sent_id]['sent_data']
        verbs = []
        
        for word_num in sent:
            base = sent[word_num]
    
            if base['upos'] == 'VERB':
                verbs.append(word_num)
                
        for word_num in sent:
            base = sent[word_num]
            head = base['head']
            xpos = base['xpos']
            deprel = base['deprel']
            
            if word_num in verbs:
                verb_number = verbs.index(word_num)
                base['arg'] = 'r' + str(verb_number) + ':root'
            else:
                if head in verbs:
                    verb_number = verbs.index(head)
                    possible_args = sent[head]['args'].keys()

                    if deprel == 'obl':
                        if xpos == 'NCFP000' and 'A0' in possible_args:
                            base['arg'] = 'r' + str(verb_number) + ':arg0'
                        elif xpos == 'NCMP000' and 'A1' in possible_args:
                            base['arg'] = 'r' + str(verb_number) + ':arg1'
                        elif xpos == 'NCFS000' or xpos == 'NCMS000' and 'A2' in possible_args:
                            base['arg'] = 'r' + str(verb_number) + ':arg2'
                        elif xpos == 'NCMS000' and 'A0' in possible_args: 
                            base['arg'] = 'r' + str(verb_number) + ':arg0'
                        else:
                            base['arg'] = '_'
                    elif deprel == 'nsubj':
                        if 'A0' in possible_args:
                            base['arg'] = 'r' + str(verb_number) + ':arg0'
                        else:
                            base['arg'] = '_'
                    elif deprel == 'obj':
                        if 'A1' in possible_args:
                            base['arg'] = 'r' + str(verb_number) + ':arg1'
                        else:
                            base['arg'] = '_'
                    elif deprel == 'iobj':
                        if 'A2' in possible_args:
                            base['arg'] = 'r' + str(verb_number) + ':arg2'
                        else:
                            base['arg'] = '_'
                    else:
                        base['arg'] = '_'
                else:
                    base['arg'] = '_'                


def treegal_to_conll_dict(file, conll_dict=None, gal_to_propbank_dict=None):
    if conll_dict==None and gal_to_propbank_dict==None:
        conll_dict = defaultdict(dict)
        gal_to_propbank_dict = defaultdict(dict)
        
    elif conll_dict==None or gal_to_propbank_dict==None:
        return 'Error! Must provide either both or neither conll_dict and gal_to_propbank_dict.'
    else:
        conll_dict = conll_dict
        gal_to_propbank_dict = gal_to_propbank_dict
            
    with open(file, 'r') as f:
        lines = f.readlines()
        sent_id = None

        for line in lines:
            if 'sent_id' in line:
                if sent_id != None:
                    conll_dict[sent_id]['sent'] = sent
                    conll_dict[sent_id]['sent_data'] = sent_data_dict

                line = line.strip().split(' ')
                sent_id = int(line[-1])
                
                sent_data_dict = defaultdict(dict)
                
                sent = None
                word_index = None
                word_form = None
                lemma = None
                upos = None
                xpos = None
                feats = None
                head = None
                deprel = None
                deps = None
                misc = None

            elif '#' in line:
                line = line.strip().split('=')
                sent = line[-1]
            else:
                line = line.split('\t')

                if len(line) < 2:
                    continue
                else:
                    word_index = line[0]
                    word_form = line[1]
                    lemma = line[2]
                    upos = line[3]
                    xpos = line[4]
                    feats = line[5]
                    head = line[6]
                    deprel = line[7]
                    deps = line[8]
                    misc = line[9].strip()

                    sent_data_dict[word_index]['word_form'] = word_form
                    sent_data_dict[word_index]['lemma'] = lemma
                    sent_data_dict[word_index]['upos'] = upos
                    sent_data_dict[word_index]['xpos'] = xpos
                    sent_data_dict[word_index]['feats'] = feats
                    sent_data_dict[word_index]['head'] = head
                    sent_data_dict[word_index]['deprel'] = deprel
                    sent_data_dict[word_index]['deps'] = deps
                    sent_data_dict[word_index]['misc'] = misc
                    sent_data_dict[word_index]['role'] = '_'
                                        
                    if upos == 'VERB':
                        # print('VERB')
                        lemma_synsets = wn.synsets(lemma, lang='glg')
                        rolesets = []
                        arg_sets = []
                                             
                        for synset in lemma_synsets:
                            split_synset = str(synset).split("'")
                            split_split = split_synset[1].split('.')
                            syn = split_split[0] + '.' + split_split[-1] #'garner.01'
                            
                            try:
                                roleset = propbank.roleset(syn)
                                rolesets.append(syn)
                                args = dict()
                                
                                for role in roleset.findall('roles/role'):
                                    arg_num = 'A' + role.attrib['n']
                                    args[arg_num] = role.attrib['descr']
                                    
                                arg_sets.append(args)
                            
                            except:
                                continue # no match
                            
                        gal_to_propbank_dict[lemma]['rolesets'] = rolesets
                        gal_to_propbank_dict[lemma]['arg_sets'] = arg_sets                          
                        
                        if len(rolesets) == 0:
                            sent_data_dict[word_index]['role'] = 'undefined.01'
                            sent_data_dict[word_index]['args'] = {'A0' : 'describer',
                                                                  'A1' : 'thing defined',
                                                                  'A2' : 'attribute'}
                        elif len(rolesets) == 1:
                            sent_data_dict[word_index]['role'] = rolesets[0]
                            sent_data_dict[word_index]['args'] = arg_sets[0]
                        
                        else:
                            sent_data_dict[word_index]['role'] = 'see gal_to_propbank_dict'
                            sent_data_dict[word_index]['args'] = {'A0' : 'describer',
                                                                  'A1' : 'thing defined',
                                                                  'A2' : 'attribute'}
                
                        
    f.close()
    return conll_dict, gal_to_propbank_dict


def treegal_conll_add_args(conll_dictionary):

    for sent_id in conll_dictionary.keys(): # each sent
        sent = conll_dictionary[sent_id]['sent_data']
        verbs = []
        
        for word_num in sent:
            base = sent[word_num]
    
            if base['upos'] == 'VERB':
                verbs.append(word_num)
                
        for word_num in sent:
            base = sent[word_num]
            head = base['head']
            xpos = base['xpos']
            deprel = base['deprel']
            
            if word_num in verbs:
                verb_number = verbs.index(word_num)
                base['arg'] = 'r' + str(verb_number) + ':root'
            else:
                if head in verbs:
                    verb_number = verbs.index(head)
                    possible_args = sent[head]['args'].keys()

                    if deprel == 'obl':
                        if xpos == 'Zgms' and 'A0' in possible_args:
                            base['arg'] = 'r' + str(verb_number) + ':arg0'
                        elif xpos == 'Scfs' or xpos == 'Tnfs' and 'A1' in possible_args:
                            base['arg'] = 'r' + str(verb_number) + ':arg1'
                        elif xpos == 'Infp' and 'A2' in possible_args:
                            base['arg'] = 'r' + str(verb_number) + ':arg2'
                        else:
                            base['arg'] = '_'
                    elif deprel == 'nsubj':
                        if 'A0' in possible_args:
                            base['arg'] = 'r' + str(verb_number) + ':arg0'
                        else:
                            base['arg'] = '_'
                    elif deprel == 'obj':
                        if 'A1' in possible_args:
                            base['arg'] = 'r' + str(verb_number) + ':arg1'
                        else:
                            base['arg'] = '_'
                    elif deprel == 'iobj':
                        if 'A2' in possible_args:
                            base['arg'] = 'r' + str(verb_number) + ':arg2'
                        else:
                            base['arg'] = '_'
                    else:
                        base['arg'] = '_'
                else:
                    base['arg'] = '_' 
                    

def write_to_conllu(conll_dict, file_name):
    with open(file_name, 'w') as f:
        f.write('# global.columns = ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC ROLE ARG\n') # ARG= root#:arg
        
        for sent_id in conll_dict:
            base = conll_dict[sent_id]['sent_data']
            f.write('\n')
            f.write(f'# sent_id = {sent_id}\n')
            f.write(f'# sent = {conll_dict[sent_id]["sent"]}\n')
            
            for word_id in base:
                word_base = base[word_id]
                f.write(f'{word_id} {word_base["word_form"]} {word_base["lemma"]} {word_base["upos"]} {word_base["xpos"]} {word_base["feats"]} {word_base["head"]} {word_base["deprel"]} {word_base["deps"]} {word_base["misc"]} {word_base["role"]} {word_base["arg"]}\n')
    f.close()
    
    
def import_data_from_conllu(file_path, ddict=None):    
    sent_count = 0
    
    if ddict:
        data_dict = ddict
    
    else:
        data_dict = {'tokens' : [],
                     'tags' : []
                    }
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        toks = []
        labels = []
        
        for line in lines:
            if '# sent ' in line:
                if sent_count != 0:
                    if toks not in data_dict['tokens']:
                        data_dict['tokens'].append(toks)
                        data_dict['tags'].append(labels)
                    toks = []
                    labels = []
                    
                sent_count += 1
                
            elif line.startswith('#') or len(line.split()) <2:
                continue
            
            else:
                line = line.split()
                tok_idx = line[0]
                tok = line[1]
                arg = line[-1]
                
                try:
                    int_tok_idx = int(tok_idx)
                    
                    toks.append(tok)
                    
                    if arg == '_':
                        labels.append('O')
                    else:
                        labels.append(arg)
                except:
                    continue
    f.close()
    return data_dict
                    

if __name__ == "__main__":
    # process CTG data & add arguments
    try:
        ctg_conll_dict_dev, ctg_gal_to_propbank_dict_dev = ctg_to_conll_dict('data/gl_ctg-ud-dev.conllu')
        ctg_conll_dict_test, ctg_gal_to_propbank_dict_test = ctg_to_conll_dict('data/gl_ctg-ud-test.conllu', 
                                                                                conll_dict=ctg_conll_dict_dev,
                                                                                gal_to_propbank_dict=ctg_gal_to_propbank_dict_dev)
        ctg_conll_dict_all, ctg_gal_to_propbank_dict_all = ctg_to_conll_dict('data/gl_ctg-ud-train.conllu',
                                                                              conll_dict=ctg_conll_dict_test,
                                                                              gal_to_propbank_dict=ctg_gal_to_propbank_dict_test)
        ctg_conll_add_args(ctg_conll_dict_all)
    except:
        print('Error! Cannot find CTG files. Please double check you have downloaded all files and that they are stored correctly.')
        print('Files should be stored as: "data/gl_ctg-ud-dev.conllu", "data/gl_ctg-ud-test.conllu", and "data/gl_ctg-ud-train.conllu".')


    # process TreeGal data & add arguments
    try:
        treegal_conll_dict_test, treegal_gal_to_propbank_dict_test = treegal_to_conll_dict('data/gl_treegal-ud-test.conllu')
        treegal_conll_dict_all, treegal_gal_to_propbank_dict_all = treegal_to_conll_dict('data/gl_treegal-ud-train.conllu',
                                                                                      conll_dict=treegal_conll_dict_test,
                                                                                      gal_to_propbank_dict=treegal_gal_to_propbank_dict_test)
        treegal_conll_add_args(treegal_conll_dict_all)
    except:
        print('Error! Cannot find TreeGal files. Please double check you have downloaded all files and that they are stored correctly.')
        print('Files should be stored: "data/gl_treegal-ud-test.conllu", and "data/gl_treegal-ud-train.conllu".') 


    # write new data dictionaries to conllu files
    write_to_conllu(ctg_conll_dict_all, 'data/ctg_conll_dict_all.conllu')
    write_to_conllu(treegal_conll_dict_all, 'data/treegal_conll_dict_all.conllu')


    # import conllu files, join CTG and TreeGal data, and remove duplicate sentences
    ctg_data = import_data_from_conllu('data/ctg_conll_dict_all.conllu')
    all_data = import_data_from_conllu('data/ctg_conll_dict_all.conllu', ddict=ctg_data) # fin_new_dict = 


    # assign sentence IDs
    all_sent_ids = [i for i in range(len(all_data['tokens']))]
    all_data['ids'] = all_sent_ids

    # ensure data combined correctly
    assert len(all_data['tags']) == len(all_data['tokens'])
    assert len(all_data['ids']) == len(all_data['tags'])


    # define labels
    all_labels = sorted(set(label for labels in all_data['tags'] for label in labels))


    # convert to Dataset object, split into train & test sets, and convert labels to ClassLabel
    gal_ds = Dataset.from_dict(all_data)
    split_gal_ds = gal_ds.train_test_split(test_size=0.2, shuffle=True)
    final_gal_ds = split_gal_ds.cast_column('tags', Sequence(ClassLabel(names=all_labels)))


    # save Dataset object locally
    final_gal_ds.save_to_disk('data/final_gal_ds.hf')

    # to load from local run command
    # final_gal_ds = load_from_disk('data/final_gal_ds.hf')