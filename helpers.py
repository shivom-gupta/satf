import collections
import boto3
from pyspark import AccumulatorParam

def create_s3_client(AWS_ACCESS_KEY_ID, AWS_SECRET, ENDPOINT_URL):
    """
    Create an S3 client using the provided AWS credentials and endpoint URL.

    :param AWS_ACCESS_KEY_ID: AWS Access Key ID
    :param AWS_SECRET: AWS Secret Access Key
    :param ENDPOINT_URL: S3 endpoint URL
    :return: S3 client object
    """
    session = boto3.session.Session(AWS_ACCESS_KEY_ID, AWS_SECRET)
    return session.client(
        service_name='s3',
        endpoint_url=ENDPOINT_URL,
    )

def get_file_stream(s3_client, file_identifier):
    """
    Retrieve the file stream from S3 given the bucket and key.

    :param s3_client: The S3 client object
    :param file_identifier: Tuple containing bucket name and key
    :return: The file stream
    """
    bucket, key = file_identifier
    response = s3_client.get_object(
        Bucket=bucket,
        Key=key
    )
    return response['Body']._raw_stream

class CounterAccumulatorParam(AccumulatorParam):
    """
    Custom AccumulatorParam class for accumulating counts using a Counter.
    """
    def zero(self, v):
        """
        Return a new, empty Counter for initializing the accumulator.

        :param v: Initial value (not used)
        :return: An empty Counter object
        """
        return collections.Counter()

    def addInPlace(self, acc1, acc2):
        """
        Add two Counter objects together in place.

        :param acc1: First Counter object
        :param acc2: Second Counter object
        :return: Combined Counter object
        """
        return acc1 + acc2
