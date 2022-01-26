# Taken from:
# https://patel-zeel.github.io/blog/data/openaq/2020/09/21/Programatically_download_OpenAQ_data.html
import pandas as pd
import boto3
import botocore
import os
from joblib import Parallel, delayed

s3 = boto3.client(
    "s3", config=botocore.config.Config(signature_version=botocore.UNSIGNED)
)
bucket_name = "openaq-fetches"
prefix = "realtime-gzipped/"

path = "./"

start_date = "2019/01/01"  # start date (inclusive)
end_date = "2019/12/31"  # end date (inclusive)


def download_date(date):
    date = str(date).split(" ")[0]  # keeping just YYYY-MM-DD from YYYY-MM-DD HH:MM:SS
    print("Downloading:", date)
    data_dict = s3.list_objects(Bucket=bucket_name, Prefix=prefix + date)

    for file_obj in data_dict["Contents"]:
        f_name = file_obj["Key"]
        tmp_path = "/".join((path + f_name).split("/")[:-1])

        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)

        s3.download_file(bucket_name, f_name, path + f_name)


# for date in pd.date_range(start=start_date, end=end_date):

Parallel(n_jobs=-1)(
    delayed(download_date(date))
    for date in pd.date_range(start=start_date, end=end_date)
)

for date in pd.date_range(start=start_date, end=end_date):
    date = str(date).split(" ")[0]  # keeping just YYYY-MM-DD from YYYY-MM-DD HH:MM:SS
    data_dict = s3.list_objects(Bucket=bucket_name, Prefix=prefix + date)

    for file_obj in data_dict["Contents"]:
        assert os.path.exists(path + file_obj["Key"]), file_obj["Key"]


print("Validated")
