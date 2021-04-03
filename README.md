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
* python3, numpy, tensorflow, pymongo, mecab, mecab-ko, mecab-python, sentencepiece

## Crawler
Collect multiple data from specific web site

example:
```
python3 main.py --op=crawler --category=휴대폰

or

python3 main.py --op=c --category=휴대폰
```

## Directory
* `util`: directory related some util files
* `util/crawler`: directory related to crawler and db controller
* `util/tokenizer`: directory related to tokenizer
* `src`: directory related to recommendation system
* `configs`: directory to with some argument
* `data`: directory where preprocessed data is located (extension is tfrecord)
* `resources`: directory with vocab files (mecab, sentencepiece, mecab-sentencepiece)