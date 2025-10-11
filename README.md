# tokenizer-filter

Tools to analyze and modify tokenizers

## Example

```
python3 filter_by_script.py openai/gpt-oss-120b Latin --save-dir harmony-latin
```

output

```
Removing 56637 tokens
Filtered vocab from 199998 to 143361
Filtered merges from 446189 to 357029
Updated ids for 21 special tokens
Updated ids for 0 special tokens in preprocessor
Saved tokenizer in harmony-latin
```

test

```
>>> from transformers import AutoTokenizer
>>> t1 = AutoTokenizer.from_pretrained('openai/gpt-oss-120b')
>>> t2 = AutoTokenizer.from_pretrained('harmony-latin')
>>> t1.tokenize('Latin unchanged') == t2.tokenize('Latin unchanged')
True
>>> t1.tokenize('日本語は違う') == t2.tokenize('日本語は違う')
False
```
