# WIP...

### run flow
``` bash
docker exec -it ml-app python3 src/train.py
```

### add minio block to save flow codes(bucket: prefect-flows/train)
ref: https://docs.prefect.io/latest/concepts/filesystems/#remote-file-system
``` bash
docker exec -it ml-app python3 deploy-prefect-flows.py
```


### create new work-pool
``` bash
docker exec -it ml-app prefect work-pool create ml-app-pool
```


### deployment app to MinIO
## Build
ref: https://docs.prefect.io/latest/concepts/deployments/
``` bash
docker exec -it ml-app \
  prefect deployment build \
    -sb remote-file-system/prefect-flows-train \
    -n training_test \
    -p ml-app-pool \
    -q train \
    src/train.py:main
```
after executing above command, train codes are pushed to minio server -> http://localhost:9001/browser/prefect-flows/train

## Apply
``` bash
docker exec -it ml-app prefect deployment apply main-deployment.yaml
```


### access to MinIO from terminal via awscli
``` bash
aws --endpoint-url http://minio:9000 s3 ...
```


https://hub.docker.com/repositories/cn47

https://docs.prefect.io/latest/api-ref/prefect/infrastructure/#prefect.infrastructure.DockerContainer


https://docs.prefect.io/latest/tutorials/deployments/
