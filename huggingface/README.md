此文件夹用于存放模型 gpt2-base

在https://huggingface.co/gpt2/tree/main上下载：

- pytorch_model.bin         (548.1mb smallest version of GPT-2, with 124M parameters.)
- config.json
- tokenizer.json
- vocab.json
- merges.txt


运行：先创建环境，安装所有所需包(如pytorch)
cd (git文件夹目录)/  (huggingface的上一级)
python test.py
python test2.py
python test3.py