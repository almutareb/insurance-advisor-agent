# scrap a given url recursively

import subprocess
import os
from urllib.parse import urlparse
from langchain_community.document_loaders import DirectoryLoader

def runcmd(cmd, verbose = False, *args, **kwargs):

    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        shell = True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass
    return process.returncode

def scrap_website(target_url:str, depth:int=5):
    target_domain = urlparse(target_url).netloc
    target_directory='./downloads/'
    # To download the files locally for processing, here's the command line
    command_this=f'wget -e robots=off --recursive -l {depth} --no-clobber --page-requisites --html-extension \
    --convert-links --restrict-file-names=windows --force-directories --directory-prefix={target_directory}\
    --domains target_domain --no-parent {target_url}'
    cmd_status = runcmd(command_this, verbose=True)
    if cmd_status==0:
        documents_path = os.path.dirname(os.path.realpath(f'{target_directory}/{target_domain}'))
        loader = DirectoryLoader(documents_path, silent_errors=True, show_progress=True)
        docs = loader.load()

    return docs