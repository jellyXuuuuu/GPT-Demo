此文件夹用于存放模型 DialoGPT-medium

在https://huggingface.co/microsoft/DialoGPT-medium 上下载：

- pytorch_model.bin         (548.1mb smallest version of GPT-2, with 124M parameters.)
- config.json
- tokenizer.json
- vocab.json
- merges.txt


运行：先创建环境，安装所有所需包(如pytorch)
cd (git文件夹目录)/  (huggingface的上一级)
python dialotest.py
