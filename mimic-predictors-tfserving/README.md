# Create warmup records

TF serving startup is slow without warmup records so create them by running:

```
python create_warmup_records.py
```

# Run locally

Running locally must be done throught Docker:

```
docker run -p 9090:9090 -t <image-name> --model_config_file=/models/model.config --port=9090 --rest_api_port=0
```

# Build Image for Huawei SWR

Make sure docker daemon is logged in to Huawei SWR first.

```
# build image
docker build -t swr.cn-north-4.myhuaweicloud.com/mimic-predictors/tf-serving:<version> .

# push image to huawei SWR
docker push swr.cn-north-4.myhuaweicloud.com/mimic-predictors/tf-serving:<version>
```
