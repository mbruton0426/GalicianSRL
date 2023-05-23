from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import load_from_disk, load_dataset
import evaluate
from seqeval.metrics import classification_report
import numpy as np

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True)
    labels = []
    
    for i, label in enumerate(examples['tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
                
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
                
            else:
                label_ids.append(-100)
                
            previous_word_idx = word_idx 
        labels.append(label_ids)
    tokenized_inputs['labels'] = labels
    
    return tokenized_inputs


def gal_compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [gal_label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [gal_label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    
    return {
        'precision': results['overall_precision'],
        'recall': results['overall_recall'],
        'f1': results['overall_f1'],
        'accuracy': results['overall_accuracy'],
    }


def set_id_label_dicts(list_labels):
    id2label = dict()
    label2id = dict()
    
    for i in range(len(list_labels)):
        id2label[i] = list_labels[i]
        label2id[list_labels[i]] = i
        
    return id2label, label2id


def final_evaluate(model, trainer, test_set):
    model.eval()
    predictions = trainer.predict(test_set)
    arg_pred = predictions.predictions
    pred_ids = arg_pred.argmax(-1)
    
    pred_ids2label = []
    gold_ids2label = []
    
    for i, pred_ary in enumerate(pred_ids):     
        label_ary = predictions.label_ids[i]
        pred_ary2list = [model.config.id2label[id] for j, id in enumerate(pred_ary) if label_ary[j] != -100]
        gold2list = [model.config.id2label[id] for id in label_ary if id != -100]

        pred_ids2label.append(pred_ary2list)
        gold_ids2label.append(gold2list)

    report = classification_report(gold_ids2label, pred_ids2label)
    for line in report.split('/n'):
        print(line)
        
        
print('gal_mBERT: model training beginning...')
        
# import Galician data
gal_ds = load_dataset('mbruton/galician_srl')
gal_label_list = gal_ds['train'].features['tags'].feature.names

tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

# Tokenize Galician Data
tokenized_gal_ds = gal_ds.map(tokenize_and_align_labels, batched=True)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
seqeval = evaluate.load('seqeval')

# Create Galician id2label/label2id
gal_id2label, gal_label2id = set_id_label_dicts(gal_label_list)
gal_num_labels = len(gal_id2label)

# Load Galician base model
gal_model = AutoModelForTokenClassification.from_pretrained(
    'bert-base-multilingual-cased', num_labels=gal_num_labels, id2label=gal_id2label, label2id=gal_label2id
)

# Set Galician model parameters & create Trainer()
gal_training_args = TrainingArguments(
    output_dir='gal_mBERT/',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=100,
    weight_decay=0.01,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    push_to_hub=False,
    report_to='none'
)

gal_trainer = Trainer(
    model=gal_model,
    args=gal_training_args,
    train_dataset=tokenized_gal_ds['train'],
    eval_dataset=tokenized_gal_ds['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=gal_compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=10)],
    
)

# Train Galician model
gal_trainer.train()

# Test Galician model
final_evaluate(gal_model, gal_trainer, tokenized_gal_ds['test'])

print('gal_mBERT: model training complete')