Code for training classifiers:

## Section 4.1-4.3

Training tokenwise classifiers:
```
python exp_01_ner_token_classification_per_layer.py --feature_layer {layer} --model {model} --learning_rate {lr} --classifier_hidden_dim {hidden_dim} --dataset {dataset} --batch_size_gen 8
```

Training adjacency classifiers:
```
python exp_04_ner_span_classification.py --model {model} --learning_rate {lr} --classifier_hidden_dim {hidden_dim} --dataset {dataset} --batch_size_train {batch_size} --batch_size_gen 8
```

Training span detection (j=k):
```
python exp_05_mention_detection.py --model {model} --learning_rate {lr} --classifier_hidden_dim {hidden_dim} --dataset {dataset} --batch_size_train {batch_size} --batch_size_gen 8
```

Training span detection (j=k-1):
```
python exp_06_mention_detection_next.py --model {model} --learning_rate {lr} --classifier_hidden_dim {hidden_dim} --dataset {dataset} --batch_size_train {batch_size} --batch_size_gen 8
```

## Section 4.4
```
python exp_08_fewshot.py --dataset {dataset} --model {model} --layer {layer} --k {k} --n_query 10 --n_episodes 200
```

## Section 4.5
```
python exp_09_mention_detection_next_generated.py --model {model} --learning_rate {lr} --classifier_hidden_dim {hidden_dim} --dataset {dataset} --batch_size_train {batch_size} --batch_size_gen 8
```