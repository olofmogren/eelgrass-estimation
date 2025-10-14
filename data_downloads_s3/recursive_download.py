import boto3
import os

def download_directory(client, bucket, path, local='/tmp'):
    """
    Downloads a directory from S3.
    """
    paginator = client.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=bucket, Delimiter='/', Prefix=path):
        # Handle subdirectories
        if result.get('CommonPrefixes'):
            for subdir in result.get('CommonPrefixes'):
                download_directory(client, bucket, subdir.get('Prefix'), local)
        # Handle files
        for file in result.get('Contents', []):
            dest_pathname = os.path.join(local, file.get('Key'))
            if not os.path.exists(os.path.dirname(dest_pathname)):
                os.makedirs(os.path.dirname(dest_pathname))
            if not file.get('Key').endswith('/'):
                print(f"Downloading {file.get('Key')} to {dest_pathname}")
                client.download_file(bucket, file.get('Key'), dest_pathname)

# --- Connection Details ---
endpoint_url = 'https://s3.ice.ri.se'
aws_access_key_id = '7KFD9V3P2AR88MC6P2I1'
aws_secret_access_key = 'hBbqdro57WAeHlFwx3PTE7RfXMcReKDDAQVAvJIe'
bucketname = 'detect-eelgrass-data'

# --- Create S3 Client ---
s3client = boto3.client('s3',
                        aws_access_key_id=aws_access_key_id,
                        aws_secret_access_key=aws_secret_access_key,
                        endpoint_url=endpoint_url)

# --- Start Download ---
# To download the entire bucket, start with an empty prefix.
download_directory(s3client, bucketname, '', local='/home/mogren/eelgrass-data')

