import pandas as pd
import geopandas as gpd
import numpy as np
import json
import os
import wget
from azure.storage.blob import ContainerClient
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

def lat_lon_to_modis_tile(lat,lon):
    """
    Get the modis tile indices (h,v) for a given lat/lon
    
    https://www.earthdatascience.org/tutorials/convert-modis-tile-to-lat-lon/
    """
    
    found_matching_tile = False
    i = 0
    while(not found_matching_tile):
        #print(i)
        found_matching_tile = lat >= modis_tile_extents[i, 4] \
        and lat <= modis_tile_extents[i, 5] \
        and lon >= modis_tile_extents[i, 2] and lon <= modis_tile_extents[i, 3]
        i += 1
        
    v = int(modis_tile_extents[i-1, 0])
    h = int(modis_tile_extents[i-1, 1])
    
    return h,v


def list_blobs_in_folder(container_name,folder_name):
    """
    List all blobs in a virtual folder in an Azure blob container
    """
    
    files = []
    generator = modis_container_client.list_blobs(name_starts_with=folder_name)
    for blob in generator:
        files.append(blob.name)
    return files
        
    
def list_hdf_blobs_in_folder(container_name,folder_name):
    """"
    List .hdf files in a folder
    """
    
    files = list_blobs_in_folder(container_name,folder_name)
    files = [fn for fn in files if fn.endswith('.hdf')]
    return files

def modis_download(identifiers, modis_container_name):

    # MODIS product id
    product = 'MOD10A1'

    dir = os.path.join('../data','mod10a1')

    # Get the horizontal and vertical ID from lat & lon
    h,v = lat_lon_to_modis_tile(identifiers[0], identifiers[1])

    folder = product + '/' + '{:0>2d}/{:0>2d}'.format(h,v) + '/' + str(identifiers[2])
    
    filenames = list_hdf_blobs_in_folder(modis_container_name,folder)
    blob_name = filenames[0]

    dir_1 = os.path.join(dir, identifiers[3], str(identifiers[2]))
    os.makedirs(dir_1, exist_ok = True)
    filename = os.path.join(dir_1, blob_name.replace('/','_'))

    url = modis_blob_root + blob_name
    if not os.path.isfile(filename):
        wget.download(url,filename, bar = False)            

if __name__ == '__main__':

    # Loading json grid_cell data
    with open('../data/grid_cells.geojson') as f:
        grid_cells = json.load(f)
    
    # Converting to geodf and getting centroid of each cell
    gdf = gpd.GeoDataFrame.from_features(grid_cells['features'])
    gdf['centroid'] = gdf['geometry'].centroid
    #print(gdf.head())

    # Loading modified train labels
    train_labels = pd.read_csv('../data/train_labels_modified.csv')

    # Code for accessing the Azure blobs for MODIS tiles, 
    # Taken from https://nbviewer.org/github/microsoft/AIforEarthDataSets/blob/main/data/modis.ipynb
    modis_account_name = 'modissa'
    modis_container_name = 'modis-006'
    modis_account_url = 'https://' + modis_account_name + '.blob.core.windows.net/'
    modis_blob_root = modis_account_url + modis_container_name + '/'

    modis_tile_extents_url = modis_blob_root + 'sn_bound_10deg.txt'

    dir = os.path.join('../data','mod10a1')
    os.makedirs(dir,exist_ok=True)
    fn = os.path.join(dir,modis_tile_extents_url.split('/')[-1])
    wget.download(modis_tile_extents_url, fn)



    # Load this file into a table, where each row is (v,h,lonmin,lonmax,latmin,latmax)
    modis_tile_extents = np.genfromtxt(fn, skip_header = 7, skip_footer = 3)

    modis_container_client = ContainerClient(account_url=modis_account_url, container_name=modis_container_name, credential=None)
    
    cell_id_list = train_labels['cell_id'].to_list()
    lat_list = train_labels['lat'].to_list()
    long_list = train_labels['long'].to_list()
    daynum_list = train_labels['daynum'].to_list()


    with mp.Pool(mp.cpu_count() - 1) as pool:
        with tqdm(total = len(train_labels), desc = 'cells') as pbar:
            temp = partial(modis_download, modis_container_name = modis_container_name)
            for i, _ in enumerate(pool.imap_unordered(temp, iterable = zip(lat_list, long_list, daynum_list, cell_id_list))):
                pbar.update()

