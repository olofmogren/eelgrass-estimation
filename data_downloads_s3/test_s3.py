from io import BytesIO
import rasterio
import boto3

endpoint_url = https://s3.ice.ri.se
aws_access_key_id = 7KFD9V3P2AR88MC6P2I1
aws_secret_access_key = hBbqdro57WAeHlFwx3PTE7RfXMcReKDDAQVAvJIe
bucketname = 'detect-eelgrass-data'

s3=boto3.resource('s3',aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, endpoint_url=endpoint_url)
s3obj = s3.Object(bucketname, key)
body = s3_tif_obj.get()['Body'].read()
filelike = BytesIO(body)
with rasterio.open(filelike) as dataset:
  print(dataset)


s3client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, endpoint_url=endpoint_url)

for obj in s3client.list_objects(Bucket=bucketname)['Contents']:
    try:
        filename = obj['Key'].rsplit('/', 1)[1]
    except IndexError:
        filename = obj['Key']

    localfilename = os.path.join('/home/mogren/eelgrass-data/', filename)  # The local directory must exist.
    s3client.download_file('mybucket', obj['Key'], localfilename)

