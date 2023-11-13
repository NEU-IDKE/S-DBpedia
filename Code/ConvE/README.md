## conve_reproduce
reproduce ConvE by pytorch without spodernet.

Paper:https://arxiv.org/abs/1707.01476.

ConvE is CNN-based model for knowlege graph Embedding, author's code implement is by spodernet which is hard understand.

We reproduce ConvE by pytorch without spodernet. It will be convenient for other researchers who are not familiar with the spodernet lib.

- If you want to implement yourself, you can refer https://github.com/TimDettmers/ConvE/issues/32.

```
-- data # 数据集存放
-- dataProcessCode # 数据处理代码文件
    -- # 代码功能体现在文件名上，代码也有部分注释
-- log # 训练测试的日志
    -- Train-数据集后缀 # 部分加test后缀的是单测试的日志
-- 其他文件是模型相关文件  

```

## 运行代码Linux命令
```
# 模板
nohup python run.py -name Train- -data DBpedia- -batch 128 > ./log/train-.log 2>&1 &

# large *
nohup python run.py -name Train-large -data DBpedia-large -batch 1024 -num_workers 7 > ./log/train-large.log 2>&1 &


# mid *
nohup python run.py -name Train-mid -data DBpedia-mid -batch 128 > ./log/train-mid.log 2>&1 &


# DPM5 *
nohup python run.py -name Train-DPM5 -data DBpedia-DPM5 -batch 128 > ./log/train-DPM5.log 2>&1 &
*

# RDM125 *
nohup python run.py -name Train-RDM125 -data DBpedia-RMD125 -batch 128 > ./log/train-RDM125.log 2>&1 &


# RDM25 * 
nohup python run.py -name Train-RDM25 -data DBpedia-RMD25 -batch 128 > ./log/train-RDM25.log 2>&1 &


# GT5E *
nohup python run.py -name Train-GT5E -data DBpedia-GT5E -batch 128 > ./log/train-GT5E.log 2>&1 &


# GT10E
nohup python run.py -name Train-GT10E -data DBpedia-GT10E -batch 128 > ./log/train-GT10E.log 2>&1 &


# GT20E *
nohup python run.py -name Train-GT20E -data DBpedia-GT20E -batch 128 > ./log/train-GT20E.log 2>&1 &


# GT50E *
nohup python run.py -name Train-GT50E -data DBpedia-GT50E -batch 128 > ./log/train-GT50E.log 2>&1 &


# halfER *
nohup python run.py -name Train-halfER -data DBpedia-halfER -batch 128 > ./log/train-halfER.log 2>&1 &

python run.py -name Train-RDM125 -data DBpedia-RDM125 -batch 128 -input_drop 0.4
```