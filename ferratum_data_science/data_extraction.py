import requests

def download_file(url):
    
    '''
    This function downloads file in default directory
    Parameters:
    url: path where file is located
    '''
    
    local_filename = url.split('/')[-1]
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
