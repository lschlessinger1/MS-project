from typing import Union

from google.cloud import storage


def upload_blob(bucket_name: Union[str, storage.Bucket],
                data: Union[str, bytes],
                destination_blob_name: str):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_string(data)

    print(f'Data uploaded to {destination_blob_name}.')

    return blob


def download_blob(bucket_name: Union[str, storage.Bucket],
                  source_blob_name: str,
                  destination_file_name: str):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print('Blob {} downloaded to {}.'.format(source_blob_name, destination_file_name))
