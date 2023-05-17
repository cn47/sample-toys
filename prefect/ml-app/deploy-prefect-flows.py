import os

from prefect.filesystems import RemoteFileSystem

minio_block = RemoteFileSystem(
    basepath="s3://prefect-flows/train",
    settings={
        "key": os.getenv("AWS_ACCESS_KEY_ID"),
        "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "client_kwargs": {
            "endpoint_url": os.getenv("AWS_ENDPOINT_URL"),
        },
    },
)
minio_block.save("prefect-flows-train", overwrite=True)
