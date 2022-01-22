from re import T
from PIL.Image import new
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
import contextily as ctx
import geopandas
import pandas as pd
import seaborn as sns
import numpy as np
import os
import gzip
import pickle
import datetime as dt
import os.path

from shapely.geometry import collection

def dataf(filename: str, verbose:bool = False) -> pd.DataFrame:
  """
  Funkcia vytvori dataframe 

  Params:
    filename: nazov suboru z ktoreho sa bude vytvarat dataframe
      (predpoklada sa pouzitie accidents.pkl.gz)

    verbose: ak True, na stdout sa vypise priebeh funkcie

  Returns:
    pd.Dataframe
  """
  if verbose:
    print(f'#\tReading file {filename}\t#')
  df = pd.read_pickle(filename)
  orig = round(df.memory_usage(deep=True).sum()/1048576,1)
  for i in df:
  
    if i == 'p2a':
      #copying date column
      df['date'] = pd.to_datetime(df[i])
    elif i == 'p1':
      #copy id
      df[i] = df[i]
    elif i == 'region':
      #copy region
      df[i] = df[i]
    elif df[i].dtypes == 'object':
      #copy object columns, replace empty as NaN, convert to categoric type
      df[i] = df[i].replace(r'^\s*$',np.NaN,regex=True).astype("category")
    else:
      #copy all other values
      df[i] = pd.to_numeric(df[i])
    
  newsize = round(df.memory_usage(deep=True).sum()/1048576,1)
  if verbose:
    print(f'>>\nOriginal size: {orig} MB\nSize after transformation: {newsize} MB')


  return df

def geof(df:pd.DataFrame) -> geopandas.GeoDataFrame:
  """ 
    Konvertovani dataframe do geopandas.GeoDataFrame se spravnym kodovanim
    Params: 
            df : pd.DataFrame -> Dataframe input

            verbose: ak true, vypisuje priebeh funkcie
    Returns:
            geopandas.GeoDataFrame -> Novy dataframe so spravnym kodovanim

    """
  geodf = pd.DataFrame()

  #delete unknown position
  df = df.dropna(subset=['d','e'])

  for i in df:
      geodf[i] = df[i]
  #geometry column
  geodf = geopandas.GeoDataFrame(geodf,geometry=geopandas.points_from_xy(geodf['d'],geodf['e']),crs="EPSG:5514")

  #convert to correct crs (from S-JTSK to WGS 84)
  geodf = geodf.to_crs(epsg=3857)


  return geodf

def map_img(gdf:geopandas.GeoDataFrame, verbose:bool = False, fig_location:str = None):
  """
  Funkcia vytvori Mapu s vyznacenim havarii zavinenych vodicmi pod vplyvom alkoholu

  Params:
    gdf: geopandas.GeoDataFrame -> geodataframe input

    verbose: ak true, vypisuje priebeh funcie

    fig_location: miesto kam sa ulozi mapa
  """

  if verbose:
    print('#\tCreating map image\t#')
  fig = plt.figure(figsize=(20,20))

  ax = fig.add_subplot()

  geodf = gdf.loc[gdf['p11'].isin([1,3,5,6,7,8,9]),['geometry']]

  geodf.plot(ax=ax,markersize=3,color='tab:red',alpha=0.5, legend=False)

  ax.set_ylim(6_200_000, 6_640_000)
  ax.set_xlim(1_340_000, 2_110_000)

  ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

  ax.axis("off")

  fig.tight_layout()
  if fig_location:
    if verbose:
      print(f'\t>Saving map to {fig_location}...')
    plt.savefig(fig_location)

def create_table(df:pd.DataFrame,verbose:bool = False):
  """
  funckia vytvori tabulku nehod na zaklade obsahu alkoholu v krvi a poctu zranenych osob

  Params:
    df: pd.DataFrame -> dataframe input

    verbose: ak True, vypisuje priebeh funkcie
  """
  if verbose:
    print('#\tPreparing Table\t#')
  
  newdf = df.loc[(df['p11'].isin([1,3,5,6,7,8,9])&(df['date'].dt.year>2019)),['p11','date','p13a','p13b','p13c']]
  newdf['celkovy pocet nehod'] = newdf['p11'].apply(lambda x:x**0)
  newdf['p11'].replace({1:'obsah alkoholu v krvi do 0,24 ‰',
                        3:'obsah alkoholu v krvi od 0,24 ‰ do 0,5 ‰',
                        5:'osoba bola pod vplyvom alkoholu a drog',
                        6:'obsah alkoholu v krvi od 0,5 ‰ do 0,8 ‰',
                        7:'obsah alkoholu v krvi od 0,8 ‰ do 1,0 ‰',
                        8:'obsah alkoholu v krvi od 1,0 ‰ do 1,5 ‰',
                        9:'obsah alkoholu v krvi 1,5 ‰ a více'
                        },inplace=True)

  newdf = newdf.groupby(['p11']).agg('sum')

  newdf.columns = ['Usmrtenych osob','tazko zranenych osob', 'lahko zranenych osob','celkovy pocet nehod']

  print(newdf.to_string(index=False))

def plot_alcohol(df:pd.DataFrame,verbose:bool = False,fig_location:str = None):
  """
  
  funckia vytvori graf nehod na zaklade poctu nehod zavinenych alkoholom v od roku 2020

  Params:
    df: pd.DataFrame -> dataframe input

    verbose: ak True, vypisuje priebeh funkcie
  
    fig_location: miesto kam sa ulozi mapa

  """
  if verbose:
    print('#\tPreparing plot\t#')

  newdf = df.loc[(df['p11'].isin([1,3,5,6,7,8,9])&(df['date'].dt.year>2019)),['p11','date']]
  newdf['count'] = newdf['p11'].apply(lambda x:x**0)
  newdf['date'] = newdf['date'].dt.month.astype('int8')
  newdf['p11'].replace({1:'obsah alkoholu v krvi do 0,24 ‰',
                        3:'obsah alkoholu v krvi od 0,24 ‰ do 0,5 ‰',
                        5:'osoba bola pod vplyvom alkoholu a drog',
                        6:'obsah alkoholu v krvi od 0,5 ‰ do 0,8 ‰',
                        7:'obsah alkoholu v krvi od 0,8 ‰ do 1,0 ‰',
                        8:'obsah alkoholu v krvi od 1,0 ‰ do 1,5 ‰',
                        9:'obsah alkoholu v krvi 1,5 ‰ a více'
                        },inplace=True)
  newdf = newdf.groupby(['date','p11']).agg('sum')
  #hue p13a lahko p13b p14c
  if verbose:
  g = sns.catplot(x='date', y='count',hue='p11',kind='bar', data=newdf.reset_index())
  (g.set_axis_labels("Mesiac","Počet nehod")._legend.set_title("Obsah alkoholu v krvi"))
  
  if fig_location:
    if verbose:
      print(f'> Saving plot to {fig_location}')

    plt.savefig(fig_location)

def stats(df:pd.DataFrame, verbose:bool = False):
  """
  funkcia spravi staistiku poctu umrti, a zraneni za priciny alkoholu za rok 2020
  
  Params:
    df : pd.Dataframe -> dataframe input
    verbose : ak True, vypise sa priebeh funkcie
  """
  if verbose:
    print("#\tGrouping data for statistics\t#")
  newdf = df.loc[(df['p11'].isin([1,3,5,6,7,8,9])&(df['date'].dt.year>2019)),['p13a','p13b','p13c']]
  cnts = pd.DataFrame()
  cnts['Pocet umrti'] = newdf['p13a']
  cnts['Pocet tazko zranenych'] = newdf['p13b']
  cnts['Pocet lahko zranenych'] = newdf['p13c']
  cnts = cnts.agg('sum')
  
  if verbose:
    print('> Data collected')
    print(cnts)

if __name__=='__main__':
  df = dataf('accidents.pkl.gz',True)
  geodf = geof(df,True)
  map_img(geodf,True,'map.png')
  create_table(df,True)
  plot_alcohol(df,verbose=True,fig_location='alc.png')
  stats(df,True)