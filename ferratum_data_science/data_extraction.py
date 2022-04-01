import requests
import sqlalchemy


def download_file(url):

    """
    This function downloads file in default directory
    Parameters:
    url: path where file is located
    """
    
    local_filename = url.split('/')[-1]
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)


def msql_conn(database, server='10.10.65.240', username='ai', password='Crown52*11'):

    """
    This function creates connection to ms sql server
    Parameters:
    database: name of database
    server: server address and port if port is not default
    username: database user name
    password: database user password
    """

    connection = sqlalchemy.create_engine('mssql+pyodbc://' + username + ':' + password + '@'
                                          + server + '/' + database + '?driver=SQL+Server')
    return connection
