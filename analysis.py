#!/usr/bin/env python3.9
# coding=utf-8
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os

# muzete pridat libovolnou zakladni knihovnu ci knihovnu predstavenou na prednaskach
# dalsi knihovny pak na dotaz

""" Ukol 1:
načíst soubor nehod, který byl vytvořen z vašich dat. Neznámé integerové hodnoty byly mapovány na -1.

Úkoly:
- vytvořte sloupec date, který bude ve formátu data (berte v potaz pouze datum, tj sloupec p2a)
- vhodné sloupce zmenšete pomocí kategorických datových typů. Měli byste se dostat po 0.5 GB. Neměňte však na kategorický typ region (špatně by se vám pracovalo s figure-level funkcemi)
- implementujte funkci, která vypíše kompletní (hlubkou) velikost všech sloupců v DataFrame v paměti:
orig_size=X MB
new_size=X MB

Poznámka: zobrazujte na 1 desetinné místo (.1f) a počítejte, že 1 MB = 1e6 B. 
"""


def _repl3(x):
    """
    Pomocna funkcia pre upravu dat vo funkcii plot_conditions
    Na zaklade podmienky danej integerovou hodnotou vracia retazec s danym typom podmienky
    """
    arr = ['neztížené','mlha','na počátku deště','déšť','sněžení','náledí','nárazový vítr']
    return arr[x-1]

def _repl2(x):
    """
    Pomocna funkcia pre upravu dat vo funkcii plot_animals
    Na zaklade danej integerovou hodnotou vracia retazec s danym typom 
    """
    arr = ["jiné","řidičem","řidičem","jiné","zvěří","jiné","jiné","jiné"]
    return arr[x]

def _repl(x):
    """
    Pomocna funkcia pre upravu dat vo funkcii roadtype
    Na zaklade danej integerovou hodnotou vracia retazec s danym typom cestnej komunikacie
    """
    arr = ["Jiná komunikace","Dvoupruhová komunikace","Třípruhová komunikace","Čtyřpruhová komunikace","Čtyřpruhová komunikace","Vícepruhová komunikace","Rychlostní komunikace"]
    return arr[x]

def get_dataframe(filename: str, verbose: bool = False) -> pd.DataFrame:
    """
    Funkcia na spracovanie dictionary s udajmi o nehodach. Datove typy sa zmensuju na najmensi mozny typ
    Parametre :
        filename = nazov suboru s datami
        verbose = boolean, ak je nastaveny True zobrazi sa originalna velkost dat a velkost dat po spracovani

    Vracia dataframe s kategorizovanymi datami pre usporu pamate
    """
    df = pd.read_pickle(filename)
    orig_size = round(sum(df.memory_usage(deep=True))/1048576,1)
    newdf=df.copy()
    for i in range(len(df.columns)):
        try:
            if df[df.columns[i]].min()<0:
                #cant be uint
                if df[df.columns[i]].max()<=127:
                    newdf[df.columns[i]] = df[df.columns[i]].astype('int8')
                elif df[df.columns[i]].max()>127 and df[df.columns[i]].max()<=32767:
                    newdf[df.columns[i]] = df[df.columns[i]].astype('int16')
                else:
                    newdf[df.columns[i]] = df[df.columns[i]].astype('int32')

            else:
                #can be uint
                if df[df.columns[i]].max()<=255:
                    newdf[df.columns[i]] = df[df.columns[i]].astype('uint8')
                elif df[df.columns[i]].max()>255 and df[df.columns[i]].max()<=65535:
                    newdf[df.columns[i]] = df[df.columns[i]].astype('uint16')
                else:
                    newdf[df.columns[i]] = df[df.columns[i]].astype('uint32')
        except:
            continue
    
    for i in range(len(df.columns)-1):
        if df[df.columns[i]].dtype == 'object':
            newdf[df.columns[i]] = df[df.columns[i]].astype('category')
    
    newdf['date'] = pd.to_datetime(df['p2a'])
    new_size = round(sum(newdf.memory_usage(deep=True))/1048576,1)
    if verbose:
        print("orig_size="+str(orig_size)+"MB")
        print("new_size="+str(new_size)+"MB")

    return newdf

# Ukol 2: počty nehod v jednotlivých regionech podle druhu silnic

def plot_roadtype(df: pd.DataFrame, fig_location: str = None,
                  show_figure: bool = False):
    """
    Funkcia na spracovanie dat 4 konkretnych regionov pre zobrazenie statistiky nehodovosti na zaklade typu komunikacie
    Parametre:
        df = dataframe s datami
        fig_location = miesto kam sa ma graf ulozit
        show_figure = ak je nastaveny na true, graf sa zobrazi
    """
    newdf = df.loc[df['region'].isin(["PHA","STC","JHM","KVK"]),['p36','p21','region']]
    newdf['p21'] = newdf['p21'].apply(_repl).astype('category')
    newdf['p36'] = newdf['p36'].apply(lambda x: x**0)

    newdf = newdf.groupby(['region','p21']).agg('sum')

    g = sns.catplot(x="region",y="p36",col='p21',kind='bar',data=newdf.reset_index(),col_wrap=3,sharey=False)
    (g.set_axis_labels("","Počet nehod").set_titles("{col_name}"))

    if show_figure:
        plt.show()
    
    if fig_location!=None:
        g.savefig(fig_location)



# Ukol3: zavinění zvěří
def plot_animals(df: pd.DataFrame, fig_location: str = None,
                 show_figure: bool = False):
    """
    Funkcia na spracovanie dat 4 konkretnych regionov pre zobrazenie statistiky nehodovosti na zaklade typu zavinenia (zviera, vodic alebo ine)
    Parametre:
        df = dataframe s datami
        fig_location = miesto kam sa ma graf ulozit
        show_figure = ak je nastaveny na true, graf sa zobrazi
    """
                 
    newdf = df.loc[df['region'].isin(["PHA","STC","JHM","KVK"]),['p36','p10','region','date']].loc[df['date'].dt.year<2021]
    newdf['p10'] = newdf['p10'].apply(_repl2).astype('category')
    newdf['p36'] = newdf['p36'].apply(lambda x: x**0)
    newdf['date'] = newdf['date'].dt.month.astype('int8')


    newdf = newdf.groupby(['region','date','p10']).agg('sum')
    
    g = sns.catplot(x="date",y='p36',col='region',hue='p10',kind='bar',data=newdf.reset_index(),col_wrap=2,sharey=False,sharex=False)
    (g.set_axis_labels("Měsíc","Počet nehod").set_titles("Kraj:{col_name}")._legend.set_title("zavinenie"))

    if fig_location!=None:
        g.savefig(fig_location)

    if show_figure:
        plt.show()


# Ukol 4: Povětrnostní podmínky
def plot_conditions(df: pd.DataFrame, fig_location: str = None,
                    show_figure: bool = False):

    """
    Funkcia na spracovanie dat 4 konkretnych regionov pre zobrazenie statistiky nehodovosti na zaklade typov poveternostnych podmienok
    Parametre:
        df = dataframe s datami
        fig_location = miesto kam sa ma graf ulozit
        show_figure = ak je nastaveny na true, graf sa zobrazi
    """
    newdf = df.loc[df['region'].isin(["PHA","STC","JHM","KVK"]),['p36','p18','region','date']].loc[df['p18']!=0].loc[df['date'].dt.year<2021]
    newdf['p36'] = newdf['p36'].apply(lambda x: x**0).astype('int8')
    newdf['p18'] = newdf['p18'].apply(_repl3).astype('category')

    newdf.pivot_table(columns="p18",values='p36',index=['date','region'],aggfunc='sum')
    newdf = newdf.pivot_table(columns="p18",values='p36',index=['date','region'],aggfunc='sum')
    stack_series = newdf.groupby([pd.Grouper(level='region'),pd.Grouper(level='date',freq="M")]).sum().stack(level='p18')
    dataf = pd.DataFrame({ "Počet nehod" : stack_series})
    sns.set_style('whitegrid')
    g = sns.relplot(x='date',y='Počet nehod',col='region',hue='p18',data=dataf,kind='line',col_wrap=2,facet_kws={'sharex':False,'sharey':False})
    (g._legend.set_title('Príčina'))
    (g.set(ylim=(0,None)))

    if fig_location!=None:
        g.savefig(fig_location)

    if show_figure:
        plt.show()

if __name__ == "__main__":
    pass
