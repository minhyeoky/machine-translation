# My DeepLearning Template

## Principles
1. Well defined entry point for train & inference
2. From raw data to trained model
3. Notebooks are not allowed to push
4. All Configurations are controlled under json file
5. Use english
6. Test wherever possible

## 
```
📦nmt
 ┣ 📂config -- configuration files for train and inference
 ┃ ┗ 📂nmt
 ┣ 📂data --   
 ┃ ┣ 📂input -- raw data for train
 ┃ ┗ 📂output -- trained models and tensorboard log files
 ┣ 📂src -- 
 ┃ ┣ 📂log -- visualization, 
 ┃ ┃ ┣ 📜__init__.py
 ┃ ┃ ┗ 📜logger.py
 ┃ ┣ 📂model
 ┃ ┃ ┣ 📜__init__.py
 ┃ ┃ ┣ 📜loss.py
 ┃ ┃ ┣ 📜metric.py
 ┃ ┃ ┗ 📜model.py
 ┃ ┣ 📂script
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜data_loader.py -- 
 ┃ ┣ 📜test.py -- entry point for inference
 ┃ ┣ 📜train.py -- entry point for train
 ┃ ┗ 📜util.py -- utility functions
 ┣ 📜.gitignore
 ┣ 📜README.md
 ┗ 📜requirements.txt
```

## Inspired from
- [pytorch-template](https://github.com/victoresque/pytorch-template/blob/master/README.md)
- [reddit](https://www.reddit.com/r/MachineLearning/comments/9xwwpd/d_how_do_you_structure_your_pytorch_deep_learning/)