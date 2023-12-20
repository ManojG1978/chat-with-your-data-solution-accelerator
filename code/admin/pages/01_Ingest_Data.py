import json
import logging
import mimetypes
import os
import sys
import traceback
import urllib.parse
from datetime import datetime, timedelta
from typing import Optional

import chardet
import requests
import streamlit as st
from azure.storage.blob import (BlobServiceClient, ContentSettings,
                                generate_blob_sas)
from dotenv import load_dotenv
from utilities.helpers.ConfigHelper import ConfigHelper

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
load_dotenv()

logger = logging.getLogger(
    'azure.core.pipeline.policies.http_logging_policy').setLevel(logging.WARNING)
st.set_page_config(page_title="Ingest Data", page_icon=os.path.join(
    'images', 'favicon.ico'), layout="wide", menu_items=None)
mod_page_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(mod_page_style, unsafe_allow_html=True)


def remote_convert_files_and_add_embeddings(process_all=False):
    """
    Converts files and adds embeddings remotely using a backend API.

    Args:
        process_all (bool, optional): Flag indicating whether to process all files. Defaults to False.

    Raises:
        Exception: If an error occurs during the API request.

    Returns:
        None
    """
    backend_url = urllib.parse.urljoin(
        os.getenv('BACKEND_URL', 'http://localhost:7071'), "/api/BatchStartProcessing")
    params = {}
    if os.getenv('FUNCTION_KEY') is not None:
        params['code'] = os.getenv('FUNCTION_KEY')
        params['clientId'] = "clientKey"
    if process_all:
        params['process_all'] = "true"
    try:
        response = requests.post(backend_url, params=params)
        if response.status_code == 200:
            st.success(
                f"{response.text}\nPlease note this is an asynchronous process and may take a few minutes to complete.")
        else:
            st.error(f"Error: {response.text}")
    except Exception:
        st.error(traceback.format_exc())


def add_urls():
    """
    Adds URLs to the system by sending a POST request to the backend API.

    This function retrieves the URLs from the session state and sends a POST request to the backend API
    for each URL. It includes the necessary parameters and JSON body in the request. If the request is
    successful, it displays a success message. If the request fails, it raises a ValueError with the
    corresponding error message.
    """
    params = {}
    if os.getenv('FUNCTION_KEY') is not None:
        params['code'] = os.getenv('FUNCTION_KEY')
        params['clientId'] = "clientKey"
    urls = st.session_state['urls'].split('\n')
    for url in urls:
        body = {
            "url": url
        }
        backend_url = urllib.parse.urljoin(
            os.getenv('BACKEND_URL', 'http://localhost:7071'), "/api/AddURLEmbeddings")
        r = requests.post(url=backend_url, params=params, json=body)
        if not r.ok:
            raise ValueError(f'Error {r.status_code}: {r.text}')
        else:
            st.success(f'Embeddings added successfully for {url}')


def upload_file(bytes_data: bytes, file_name: str, content_type: Optional[str] = None):
    """
    Uploads a file to Azure Blob Storage.

    Args:
        bytes_data (bytes): The file data in bytes format.
        file_name (str): The name of the file.
        content_type (str, optional): The content type of the file. If not provided, it will be guessed based on the file extension.

    Raises:
        ValueError: If the required environment variables are not set.

    Returns:
        None
    """

    # Upload a new file
    st.session_state['filename'] = file_name
    if content_type is None:
        content_type = mimetypes.MimeTypes().guess_type(file_name)[0]
        charset = f"; charset={chardet.detect(bytes_data)['encoding']}" if content_type == 'text/plain' else ''
        content_type = content_type if content_type is None else 'text/plain'
    account_name = os.getenv('AZURE_BLOB_ACCOUNT_NAME')
    account_key = os.getenv('AZURE_BLOB_ACCOUNT_KEY')
    container_name = os.getenv('AZURE_BLOB_CONTAINER_NAME')
    if account_name is None or account_key is None or container_name is None:
        raise ValueError(
            "Please provide values for AZURE_BLOB_ACCOUNT_NAME, AZURE_BLOB_ACCOUNT_KEY and AZURE_BLOB_CONTAINER_NAME")
    connect_str = f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"
    blob_service_client: BlobServiceClient = BlobServiceClient.from_connection_string(
        connect_str)
    # Create a blob client using the local file name as the name for the blob
    blob_client = blob_service_client.get_blob_client(
        container=container_name, blob=file_name)
    # Upload the created file
    blob_client.upload_blob(bytes_data, overwrite=True, content_settings=ContentSettings(
        content_type=content_type+charset))
    # Generate a SAS URL to the blob and return it
    st.session_state['file_url'] = blob_client.url + '?' + generate_blob_sas(
        account_name, container_name, file_name, account_key=account_key,  permission="r", expiry=datetime.utcnow() + timedelta(hours=3))


try:
    with st.expander("Add documents in Batch", expanded=True):
        config = ConfigHelper.get_active_config_or_default()
        file_type = [
            processor.document_type for processor in config.document_processors]
        uploaded_files = st.file_uploader(
            "Upload a document to add it to the Azure Storage Account, compute embeddings and add them to the Azure AI Search index. Check your configuration for available document processors.", type=file_type, accept_multiple_files=True)
        if uploaded_files is not None:
            for up in uploaded_files:
                # To read file as bytes:
                bytes_data = up.getvalue()
                if st.session_state.get('filename', '') != up.name:
                    # Upload a new file
                    upload_file(bytes_data, up.name)
            if len(uploaded_files) > 0:
                st.success(f"{len(uploaded_files)} documents uploaded. Embeddings computation in progress. \nPlease note this is an asynchronous process and may take a few minutes to complete.\nYou can check for further details in the Azure Function logs.")

        col1, col2, col3 = st.columns([2, 1, 2])
        with col3:
            st.button("Reprocess all documents in the Azure Storage account",
                    on_click=remote_convert_files_and_add_embeddings, args=(True,))

    with st.expander("Add URLs to the knowledge base", expanded=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.text_area("Add a URLs and than click on 'Compute Embeddings'",
                        placeholder="PLACE YOUR URLS HERE SEPARATED BY A NEW LINE", height=100, key="urls")

        with col2:
            st.selectbox('Embeddings models', [os.getenv(
                'AZURE_OPENAI_EMBEDDING_MODEL')], disabled=True)
            st.button("Process and ingest web pages",
                    on_click=add_urls, key="add_url")

except Exception:
    st.error(traceback.format_exc())
