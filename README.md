# Test Locally

Install all requirements listed in `requirements.txt` including `uvicorn` and `fastapi` and run the following command:

```
cd app
uvicorn main:app --reload
```

Make sure that `tf-serving` and mongodb is running.

# Build Image for Huawei SWR

Make sure docker daemon is logged in to Huawei SWR first.

```bash
# build image
docker build -t swr.cn-north-4.myhuaweicloud.com/mimic-predictors/backend:<version> .

# push image to huawei SWR
docker push swr.cn-north-4.myhuaweicloud.com/mimic-predictors/backend:<version>
```

# Apply rolling update to Docker Swarm stack

Make sure docker daemon is logged in to Huawei SWR first.

```
docker service update --image swr.cn-north-4.myhuaweicloud.com/mimic-predictors/backend:<version> <docker-stack-backend-service-name> --with-registry-auth
```
