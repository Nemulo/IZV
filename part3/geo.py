#!/usr/bin/python3.8
# coding=utf-8
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import contextily
from shapely.geometry import geo
import sklearn.cluster
import numpy as np
# muzete pridat vlastni knihovny


def make_geo(df: pd.DataFrame) -> geopandas.GeoDataFrame:
    """ 
    Konvertovani dataframe do geopandas.GeoDataFrame se spravnym kodovanim
    Params: 
            df : pd.DataFrame -> Dataframe input
    Returns:
            geopandas.GeoDataFrame -> Novy dataframe so spravnym kodovanim

    """
    
    geodf = pd.DataFrame()

    #delete unknown position
    df = df.dropna(subset=['d','e'])

    for i in df:
        if i == 'p2a':
            #copying date column
            geodf['date'] = pd.to_datetime(df[i])
        elif i == 'p1':
            #copy id
            geodf[i] = df[i]
        elif i == 'region':
            #copy region
            geodf[i] = df[i]
        elif df[i].dtypes == 'object':
            #copy object columns, replace empty as NaN, convert to categoric type
            geodf[i] = df[i].replace(r'^\s*$',np.NaN,regex=True).astype("category")
        else:
            #copy all other values
            geodf[i] = pd.to_numeric(df[i])

    #geometry column
    geodf = geopandas.GeoDataFrame(geodf,geometry=geopandas.points_from_xy(geodf['d'],geodf['e']),crs="EPSG:5514")

    #convert to correct crs (from S-JTSK to WGS 84)
    geodf = geodf.to_crs(epsg=3857)

    return geodf


def plot_geo(gdf: geopandas.GeoDataFrame, fig_location: str = None,
             show_figure: bool = False):
    """ 
    Vykresleni grafu s sesti podgrafy podle lokality nehody 
     (dalnice vs prvni trida) pro roky 2018-2020 
    Params:
        gdf: geopandas.GeoDataFrame -> geodataframe input

        fig_location: nazov suboru kam sa ulozia grafy

        show_figure: ak True, grafy sa zobrazia
    """
    # Select needed columns and rows

    gdf = gdf.loc[(gdf['region'].isin(["JHM"])) & (gdf['date'].dt.year.isin([2018,2019,2020])),['p36','date','geometry']]

    # Prepare figure, grid, and list of axes
    fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(nrows=3,ncols=2,figsize=(10,15),layout='constrained')
    gdf[(gdf['p36'] == 0) & (gdf['date'].dt.year == 2018)].plot(ax=ax1,markersize=3,color='tab:green')
    gdf[(gdf['p36'] == 0) & (gdf['date'].dt.year == 2019)].plot(ax=ax3,markersize=3,color='tab:green')
    gdf[(gdf['p36'] == 0) & (gdf['date'].dt.year == 2020)].plot(ax=ax5,markersize=3,color='tab:green')
    
    gdf[(gdf['p36'] == 1) & (gdf['date'].dt.year == 2018)].plot(ax=ax2,markersize=3,color='tab:red')
    gdf[(gdf['p36'] == 1) & (gdf['date'].dt.year == 2019)].plot(ax=ax4,markersize=3,color='tab:red')
    gdf[(gdf['p36'] == 1) & (gdf['date'].dt.year == 2020)].plot(ax=ax6,markersize=3,color='tab:red')
    
    ax1.set_title("PHA kraj: dialnica(2018)")
    ax3.set_title("PHA kraj: dialnica(2019)")
    ax5.set_title("PHA kraj: dialnica(2020)")

    ax2.set_title("PHA kraj: cesta prvej triedy(2018)")
    ax4.set_title("PHA kraj: cesta prvej triedy(2019)")
    ax6.set_title("PHA kraj: cesta prvej triedy(2020)")
    for i in [ax1,ax2,ax3,ax4,ax5,ax6]:
            i.set_ylim(6_205_000, 6_390_000)
            i.set_xlim(1_725_000, 1_972_500)
            contextily.add_basemap(i, source=contextily.providers.Stamen.TonerLite)
            i.axis('off')

    fig.align_labels()

    if fig_location:
        plt.savefig(fig_location)

    
    if show_figure:
        plt.show()                              
    

def plot_cluster(gdf: geopandas.GeoDataFrame, fig_location: str = None,
                 show_figure: bool = False):
    """ 
    Vykresleni grafu s lokalitou vsech nehod v kraji shlukovanych do clusteru 
    
    Params:
        gdf: geopandas.GeoDataFrame -> geodataframe input

        fig_location: nazov suboru kam sa ulozia grafy

        show_figure: ak True, grafy sa zobrazia

    Pouzitie clusteringu na zaklade metody Kmeans
    """

    gdf = gdf.loc[(gdf['region'].isin(["JHM"])&(gdf['p36']==1)),['geometry']]

    coords = np.dstack([gdf.geometry.x, gdf.geometry.y]).reshape(-1, 2)
    
    db = sklearn.cluster.MiniBatchKMeans(n_clusters=25).fit(coords)
    
    gdf4 = gdf.copy()
    
    gdf4["cluster"] = db.labels_

    gdf4['count'] = gdf4['cluster'].apply(lambda x:x**0)

    gdf4 = gdf4.dissolve(by="cluster", aggfunc={"count": "sum"})

    fig = plt.figure(figsize=(20, 20)) 

    ax = plt.gca()

    gdf4.plot(ax=ax, alpha=0.5, column='count', markersize=3, legend=True)

    ax.set_ylim(6_205_000, 6_390_000)
    ax.set_xlim(1_725_000, 1_972_500)

    ax.axis('off')

    ax.set_title("Nehody v JHM kraji na cestach 1. triedy")

    contextily.add_basemap(ax, crs=gdf.crs.to_string(), source=contextily.providers.Stamen.Terrain, alpha=0.6)
    
    if fig_location:
        plt.savefig(fig_location)

    if show_figure:
        plt.show()
    


if __name__ == "__main__":
    # zde muzete delat libovolne modifikace
    gdf = make_geo(pd.read_pickle('accidents.pkl.gz'))
    plot_geo(gdf,'geo1.png',True)
    plot_cluster(gdf,'geo1.png',True)
    
