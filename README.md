# Neural Machine Translation

## Objectives
1. Make flexible deeplearning project template 
2. Compare performances of different models & methods
3. Understand and implement attention perfectly

## Models
- [x] Seq2seq
- [ ] Attention
- [ ] Transformer

## Visualization
- [x] BLEU [train, test]
- [x] Losses [train]
- [x] Times per step
- [x] Text examples [train, test]
  
## Principles
1. Well defined entry point for train & inference
2. From raw data to trained model
3. Notebooks are not allowed to push
4. All Configurations are controlled under json file
5. Use english
6. Test wherever possible

## Directory structure
```
📦nmt
 ┣ 📂config
 ┣ 📂data
 ┃ ┣ 📂input
 ┃ ┗ 📂output
 ┣ 📂script
 ┣ 📂src
 ┃ ┣ 📂data
 ┃ ┣ 📂log
 ┃ ┣ 📂model
 ┃ ┣ 📂preprocessor
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜config.py
 ┃ ┗ 📜utils.py
 ┣ 📂tests
 ┣ 📜.gitignore
 ┣ 📜README.md
 ┣ 📜requirements.txt
 ┣ 📜run.sh
 ┗ 📜train.py
```

## References
- [Tensorflow-tutorials](https://www.tensorflow.org/tutorials/text/nmt_with_attention)
- [huggingface](https://github.com/huggingface/transformers)