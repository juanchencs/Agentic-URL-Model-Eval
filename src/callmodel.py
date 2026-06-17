import logging
import json
import boto3
from botocore.exceptions import ClientError
import argparse
import time
import base64
import os
import random
import requests
from requests.exceptions import RequestException
from datetime import datetime


BEDROCK_ROLE_ARN = os.getenv("BEDROCK_ROLE_ARN", "")
BEDROCK_EXTERNAL_ID = os.getenv("BEDROCK_EXTERNAL_ID", "")
BEDROCK_AWS_ACCESS_KEY_ID = os.getenv("BEDROCK_AWS_ACCESS_KEY_ID", "")
BEDROCK_AWS_SECRET_ACCESS_KEY = os.getenv("BEDROCK_AWS_SECRET_ACCESS_KEY", "")
 


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

_REFRESH_MARGIN_S = 300  # refresh 5 min before expiry
_bedrock_client = None
_bedrock_client_expiry = 0.0

# ---------------------------------------------------------------------------
AWS_REGION = os.getenv("AWS_REGION", "eu-west-2")

def _get_sts_client():
    """Return an STS client using dedicated Bedrock credentials from config."""
    if BEDROCK_AWS_ACCESS_KEY_ID and BEDROCK_AWS_SECRET_ACCESS_KEY:
        return boto3.client(
            "sts",
            region_name=AWS_REGION,
            aws_access_key_id=BEDROCK_AWS_ACCESS_KEY_ID,
            aws_secret_access_key=BEDROCK_AWS_SECRET_ACCESS_KEY,
        )
    return boto3.client("sts", region_name=AWS_REGION)


def _assume_bedrock_role() -> dict:
    """Assume the dedicated Bedrock IAM role via STS and return temp credentials."""
    sts = _get_sts_client()

    tags = [
        {"Key": "Project", "Value": "modelevalulation"},
        {"Key": "RunDate", "Value": datetime.utcnow().strftime("%Y-%m-%d")},
    ]
    tags.extend([])

    kwargs = {
        "RoleArn": BEDROCK_ROLE_ARN,
        "RoleSessionName": "example-model-evaluation",
        "Tags": tags,
        "DurationSeconds": 3600,
    }
    if BEDROCK_EXTERNAL_ID:
        kwargs["ExternalId"] = BEDROCK_EXTERNAL_ID

    logger.info("Assuming role %s for Bedrock access", BEDROCK_ROLE_ARN)
    response = sts.assume_role(**kwargs)
    return response["Credentials"]


def get_bedrock_client():
    """
    Return a bedrock-runtime client, refreshing STS credentials before expiry.

    Authentication priority:
    1. BEDROCK_ROLE_ARN set: use STS AssumeRole with auto-refresh.
    2. BEDROCK_AWS_ACCESS_KEY_ID set (no role): use the keys directly (cached forever).
    3. Neither set: fall back to default boto3 credential chain (cached forever).
    """
    global _bedrock_client, _bedrock_client_expiry

    if BEDROCK_ROLE_ARN:
        if _bedrock_client is None or time.monotonic() >= _bedrock_client_expiry:
            creds = _assume_bedrock_role()
            _bedrock_client = boto3.client(
                "bedrock-runtime",
                region_name='eu-west-2',
                aws_access_key_id=creds["AccessKeyId"],
                aws_secret_access_key=creds["SecretAccessKey"],
                aws_session_token=creds["SessionToken"],
            )
            _bedrock_client_expiry = time.monotonic() + 3600 - _REFRESH_MARGIN_S
            logger.info("Bedrock client created/refreshed (next refresh in ~55 min)")
        return _bedrock_client

    if _bedrock_client is not None:
        return _bedrock_client

    if BEDROCK_AWS_ACCESS_KEY_ID and BEDROCK_AWS_SECRET_ACCESS_KEY:
        logger.info("Using dedicated Bedrock access keys (no AssumeRole)")
        _bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=AWS_REGION,
            aws_access_key_id=BEDROCK_AWS_ACCESS_KEY_ID,
            aws_secret_access_key=BEDROCK_AWS_SECRET_ACCESS_KEY,
        )
    else:
        logger.info("No Bedrock credentials configured, using default chain")
        _bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)

    _bedrock_client_expiry = float("inf")
    return _bedrock_client

 