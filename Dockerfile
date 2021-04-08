FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8-slim

# pip mirror to use (choose fast mirr)
# ARG PIP_MIRROR=https://pypi.python.org/simple/
ARG PIP_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple  

COPY requirements.txt /
RUN python -m pip install --upgrade pip
RUN pip install --upgrade pip -i $PIP_MIRROR
RUN pip install -r /requirements.txt -i $PIP_MIRROR
Run pip install install torch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0
# RUN pip install torch==1.7.0+cpu torchvision==0.8.0+cpu torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
# Run pip install torch==1.7.0+cu110 torchvision==0.8.0+cu110 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

COPY ./app /app
