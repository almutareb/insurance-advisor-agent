from pathlib import Path
import boto3
from botocore.client import Config
from botocore import UNSIGNED
from dotenv import load_dotenv
import os
import sys
import zipfile


S3_LOCATION = os.getenv("S3_LOCATION")


def download_chroma_from_s3(s3_location:str,
                            chroma_vs_name:str,
                            vectorstore_folder:str,
                            vs_save_name:str) -> None:
    """
    Downloads the Chroma DB from an S3 storage to local folder
        
        Args
            s3_location (str): The name of S3 bucket
            chroma_vs_name (str): The name of the file to download from S3
            vectorstore_folder (str): The filepath to vectorstore folder in project dir
            vs_save_name (str): The name of the vector store

    """
    vs_destination = Path()/vectorstore_folder/vs_save_name
    vs_save_path = vs_destination.with_suffix('.zip')

    try:
        # Initialize an S3 client with unsigned configuration for public access
        s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        s3.download_file(s3_location, chroma_vs_name, vs_save_path)

        # Extract the zip file
        with zipfile.ZipFile(file=str(vs_save_path), mode='r') as zip_ref:
            zip_ref.extractall(path=vectorstore_folder)
        
    except Exception as e:
        print(f"Error during downloading or extracting from S3: {e}", file=sys.stderr)

    # Delete the zip file
    vs_save_path.unlink()

if __name__ == "__main__":
    chroma_vs_name = "vectorstores/chroma-zurich-mpnet-1500.zip"
    project_dir = Path().cwd().parent
    vs_destination = str(project_dir / 'vectorstore')
    assert Path(vs_destination).is_dir(), "Cannot find vectorstore folder"

    download_chroma_from_s3(s3_location=S3_LOCATION, 
                            chroma_vs_name=chroma_vs_name, 
                            vectorstore_folder=vs_destination,
                            vs_save_name='chroma-zurich-mpnet-1500')