# Product recommendation system

## Model

> **ALBERT: A Lite BERT for Self-supervised Learning of Language Representations** [[pdf](https://arxiv.org/abs/1909.11942)]<br>
> [Zhenzhong Lan](google)\*, [Mingda Chen](ttic)\*, [Sebastian Goodman](google)\*, [Kevin Gimpel](ttic)\*, [Piyush Sharma](google)\*, [Radu Soricut](google)\*<br>
> Accepted to AACL-IJCNLP 2020. (*indicates equal contribution)

> **ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators** [[pdf](https://arxiv.org/abs/2003.10555)]<br>
> [Kevin Clark](Standford University)\*, [Minh-Tang Luong](Google Brain)\*, [Quoc V. Le](Google Brain)\*, [Christopher D. Manning](Google Brain)\*<br>
> Accepted to AACL-IJCNLP 2020. (*indicates equal contribution)


## Installation
```
pip install -r requirements.txt
```

## Dependency
* python3.6, numpy, tensorflow, pymongo, mecab, mecab-ko, mecab-python, sentencepiece, rich

## Crawler
Collect multiple data from specific web site

example:
```
python3 main.py --op=crawler --category=cellphone
```

## Create pretrain data
example:
```
python3 main.py --op=create_pretrain \
                --input_file=file1, file2, file3 \
                --output_file=file1.tfrecord, file2.tfrecord, file3.tfrecord \
                --vocab_file=resources/mecab_sp-32k/tok.vocab \
                --spm_model=resources/mecab_sp-32k/tok.model \
                --mecab_file=resources/mecab_sp-32k/tok.json \
                --parallel=True
```

## Train pretrain model
example:
```
python3 main.py --op=pretrain \
                --data_home_dir=data \
                --checkpoint_dir=checkpoint \
                --tokenizer_dir=resources/mecab_sp-32k \
                --config_file=config/electra_albert_base.json
```

## Create pretrain data
example:
```
python3 main.py --op=create_finetune
```

## Directory
* `util`: directory related some util files
* `util/crawler`: directory related to crawler and db controller
* `util/tokenizer`: directory related to tokenizer
* `src`: directory related to recommendation system
* `configs`: directory to with some argument
* `data`: directory where preprocessed data is located (extension is tfrecord)
* `resources`: directory with vocab files (mecab, sentencepiece, mecab-sentencepiece)

## License
Apache License 2.0