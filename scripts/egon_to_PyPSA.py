import pandas as pd
import geopandas as gpd
import pypsa
from pypsa.clustering.spatial import (
    aggregatebuses,
    #aggregategenerators,
    aggregateoneport,
    get_clustering_from_busmap,
)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
from pypsa.geo import haversine_pts
import networkx as nx
from networkx.algorithms.community.quality import modularity
from scipy.sparse.csgraph import connected_components, dijkstra
import numpy as np
import shapely
from shapely import wkb
import shapely.wkt
from shapely.geometry import mapping
from shapely.geometry import Point
import warnings
import atlite
from dask.distributed import Client
import re
from shapely.geometry import LineString as Line
import xarray as xr
#from _helpers import REGION_COLS
#matplotlib.use('Qt5Agg')
matplotlib.use('TkAgg')

def base_from_egon(countries,country_shape,offshore_shape):
    scenario = 'status2019_PoWerD_v3'
    folder_name='D:/Ablage/mast/PythonSkripts/eGo_data/eTraGo_09/'+scenario

    # pypsa_version = 'pypsa-eur_v2025.07' #pypsa-eur3
    # resources = ["TM-elec4"] #["TM-EHV","TM-EHV+HV"]
    # countries = ['AT', 'BE', 'CH', 'CZ', 'DE', 'DK', 'FR', 'GB', 'LU', 'NL', 'NO', 'PL', 'SE'] #['AT', 'CH', 'CZ', 'DE', 'DK', 'FR', 'LU', 'NL', 'PL', 'SE']
    # country_shape="C:/" + pypsa_version + "/resources/"+resources[0]+"/country_shapes.geojson"
    # offshore_shape="C:/" + pypsa_version + "/resources/"+resources[0]+"/offshore_shapes.geojson"
    dynamic_line_rating = False

    #base

    b0 = pypsa.Network(folder_name + '/etrago_'+scenario+'_elec.nc')
    b1 = b0.copy()

    if 'eTraGo' in folder_name:
        b1.buses.loc[['30049', '30047'],'v_nom'] = 110.0
        b1.lines.loc['6391','v_nom'] = 110.0
        b1.lines.loc[(b1.lines.bus0=='13702'),'bus1'] = '29169' # OWP Riffgat 110kV -> UW Emden/Borssum 110kV
        b1.buses.loc[['13798', '29729'],['y','x','country']] = [48.51549841710694, 13.706436915453663, 'AT'] # Koordinaten Kraftwerk Jochstein nach Österreich versetzt
        # ergänzt Trafo Siems 380/220kV
        new_trafo = pd.DataFrame(columns=b1.transformers.replace('',np.nan).dropna(how='all',axis=1).columns)
        new_trafo.loc[str(b0.transformers.index.astype('int').max()+1),['bus0', 'bus1','s_nom', 'x']] = ['32987', '30020', 2000.0, 0.000068]
        new_trafo.loc[:,['model', 'r', 'g', 'b','num_parallel', 's_nom_max', 's_max_pu','s_nom_extendable','s_nom_mod',
                         'capital_cost', 'tap_ratio', 'tap_side', 'tap_position','phase_shift', 'active', 'build_year', 'lifetime', 'v_ang_min',
                         'v_ang_max', 'x_pu', 'r_pu', 'g_pu', 'b_pu', 'x_pu_eff','r_pu_eff','s_nom_opt', 'scn_name']]=['t', 0.0,  0.0,  0.0, 1.0, np.inf, 1.0, True, 0.0, 776.847888, 1.0, 0, 0, 0.0, True, 0, 40.0, -np.inf, np.inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 'status2019']
        b1.add("Transformer", new_trafo.index, **new_trafo)
        b1.transformers["s_nom_min"] = b1.transformers["s_nom"]
        b1.buses.loc[b1.buses.country=='','country'] = 'DE'
        delete_buses = b1.buses.loc[~b1.buses.country.isin(countries)].index
        b1.remove("Bus", delete_buses)
        # #_remove_dangling_branches
        dangling_lines = b1.lines.loc[~(b1.lines.bus0.isin(b1.buses.index) & b1.lines.bus1.isin(b1.buses.index))].index.tolist()
        dangling_links = b1.links.loc[~(b1.links.bus0.isin(b1.buses.index) & b1.links.bus1.isin(b1.buses.index))].index.tolist()
        dangling_trafos = b1.transformers.loc[~(b1.transformers.bus0.isin(b1.buses.index) & b1.transformers.bus1.isin(b1.buses.index))].index.tolist()
        b1.remove("Line", dangling_lines)
        b1.remove("Link", dangling_links)
        b1.remove("Transformer", dangling_trafos)
        b1.lines.loc[b1.lines.cables.isna(),'cables'] = b1.lines.loc[b1.lines.cables.isna(),'num_parallel'].round(0).mul(3)
        b1.lines.num_parallel = b1.lines.cables.div(3).astype('int32')
        r_per_length = b1.lines['r'] / b1.lines['length']
        x_per_length = b1.lines['x'] / b1.lines['length']
        i_nom = b1.lines["s_nom"] / (np.sqrt(3) * b1.lines["v_nom"] * b1.lines["num_parallel"])
        b1.lines['r_per_length'] = r_per_length
        b1.lines['x_per_length'] = x_per_length
        b1.lines["i_nom"] = i_nom
        b1.lines.type = ''
        print('LineType_380: ') 
        print(b1.lines.loc[b1.lines.v_nom==380,['r_per_length', 'x_per_length','i_nom']].mean())
        b1.lines.drop(['cables', 'scn_name','r_per_length', 'x_per_length','i_nom'], axis=1, errors="ignore", inplace=True)
        def _set_links_underwater_fraction(n, offshore_shapes):
            if n.links.empty:
                return
        
            if not hasattr(n.links, "geom"):
                n.links["underwater_fraction"] = 0.0
            else:
                offshore_shape = gpd.read_file(offshore_shapes).union_all()
                #links = gpd.GeoSeries(n.links.geom.dropna().map(shapely.wkt.loads))
                n.links['geometry'] = n.links['geom'].apply(wkb.loads, hex=True)
                links = gpd.GeoSeries(n.links.geometry,crs=4326)
                n.links["underwater_fraction"] = (
                    links.intersection(offshore_shape).length / links.length
                )
        # n.links['geometry'] = n.links['geom'].apply(wkb.loads, hex=True)
        # links=gpd.GeoDataFrame(n.links,geometry="geometry",crs=4326)
        _set_links_underwater_fraction(b1, offshore_shape)
        b1.links.drop(['geom', 'scn_name', 'topo'], axis=1, errors="ignore", inplace=True)
        trafo_buses = set(b1.transformers.bus0)|set(b1.transformers.bus1)
        gen_buses = set(b1.generators.bus)
        load_buses = set(b1.loads.bus)
        link_buses = set(b1.links.bus0)|set(b1.links.bus1)
        b1.buses.loc[b1.buses.index.isin(list(trafo_buses|gen_buses|load_buses|link_buses)),"symbol"]='substation'
        b1.buses.loc[b1.buses.symbol.isna(),"symbol"] = '-'
        buses = b1.buses
        def buses_in_shape(shape):
            shape = shapely.prepared.prep(shape)
            return pd.Series(
                np.fromiter(
                    (
                        shape.contains(Point(x, y))
                        for x, y in buses.loc[:, ["x", "y"]].values
                    ),
                    dtype=bool,
                    count=len(buses),
                ),
                index=buses.index,
            )
        country_shapes = gpd.read_file(country_shape).set_index("name")["geometry"]
        # reindexing necessary for supporting empty geo-dataframes
        offshore_shapes = gpd.read_file(offshore_shape)
        offshore_shapes = offshore_shapes.reindex(columns=["name", "geometry"]).set_index(
            "name"
        )["geometry"]
        substation_b = buses["symbol"].str.contains(
            "substation|converter station", case=False
        )
        def prefer_voltage(x, which):
            index = x.index
            if len(index) == 1:
                return pd.Series(index, index)
            key = (
                x.index[0]
                if x["v_nom"].isnull().all()
                else getattr(x["v_nom"], "idx" + which)()
            )
            return pd.Series(key, index)

        compat_kws = {}
        gb = buses.loc[substation_b].groupby(
            ["x", "y"], as_index=False, group_keys=False, sort=False
        )
        bus_map_low = gb.apply(prefer_voltage, "min", **compat_kws)
        lv_b = (bus_map_low == bus_map_low.index).reindex(buses.index, fill_value=False)
        bus_map_high = gb.apply(prefer_voltage, "max", **compat_kws)
        hv_b = (bus_map_high == bus_map_high.index).reindex(buses.index, fill_value=False)

        onshore_b = pd.Series(False, buses.index)
        offshore_b = pd.Series(False, buses.index)
        for country in countries:
            onshore_shape = country_shapes[country]
            onshore_country_b = buses_in_shape(onshore_shape)
            onshore_b |= onshore_country_b
            if country not in offshore_shapes.index:
                continue
            offshore_country_b = buses_in_shape(offshore_shapes[country])
            offshore_b |= offshore_country_b
        offshore_wind = pd.Series(False, buses.index)
        offshore_wind.loc[offshore_wind.index.isin(list(set(b1.generators.loc[b1.generators.carrier=='wind_offshore','bus'])))]=True
        # offshore_wind.loc['13079']=True # OWP Nordergründe 110kV
        # offshore_wind.loc['28833']=False # UW Inhaus; OWP Nordergründe 110kV
        # offshore_wind.loc['13702']=True # OWP Riffgat 110kV
        offshore_wind.loc['29170']=False # UW Emden/Borssum 220kV; OWP Riffgat 110kV
        offshore_wind.loc['29169']=True # UW Emden/Borssum 110kV; OWP Riffgat 110kV
        # offshore_wind.loc['13868']=True # OWP Alpha Ventus 110kV
        # offshore_wind.loc['30056']=False # UW Hagermarsch; OWP Alpha Ventus 110kV
        onshore_b.loc[onshore_b.index.isin(list(trafo_buses|load_buses|link_buses))]=True
        onshore_b.loc[onshore_b.index.isin(set(b1.buses[b1.buses.country=='DE'].index)-set(b1.generators.loc[b1.generators.carrier=='wind_offshore','bus']))]=True
        onshore_b.loc[onshore_b.index.isin(set(b1.generators['bus'])-set(b1.generators.loc[b1.generators.carrier=='wind_offshore','bus']))]=True
        b1.buses.loc[:,"onshore_bus"] = onshore_b
        b1.buses.loc[:,"substation_lv"] = ((lv_b & onshore_b)| offshore_wind)
        # b1.buses.loc[:,"substation_off"] = (offshore_b | (offshore_wind) |(hv_b & onshore_b))
        b1.buses.loc[:,"substation_off"] = ((offshore_wind))
        b1.buses.loc[((b1.buses.index.isin(list(trafo_buses)))&(b1.buses.substation_lv==False)&(b1.buses.substation_off==False)),['substation_lv','onshore_bus']] = [True, True]
        b1.buses.loc['29128','substation_lv'] = True # Kraftwerk Rostock (Steinkohle)
        b1.buses.drop(["geometry","tags",'scn_name', 'geom','symbol'], axis=1, inplace=True, errors="ignore")
        ####
        # test = b1.buses[["x", "y","substation_off","substation_lv","country","v_nom"]].copy()
        # test[["x", "y"]] = test[["x", "y"]].round(6)
        # trafo_buses = test[test[["x", "y"]].duplicated(keep=False)].index.tolist()
        # df = test.loc[trafo_buses,["x", "y","substation_off","substation_lv","v_nom"]].groupby(["x", "y"]).agg({
        #     "substation_off": "any",  # Mindestens ein True -> True
        #     "substation_lv": "any",   # Mindestens ein True -> True
        #     "v_nom": lambda x: not np.all(np.isclose(x, x.iloc[0], atol=1e-6))  # True, wenn sich Werte unterscheiden
        # }).reset_index() #.any()
        # df = df[((df["substation_off"]) | (df["substation_lv"]))  # Mindestens eine Spalte ist True
        #     & (df["v_nom"])  # v_nom muss True sein
        # ]
        # df['trafo'] = True
        # result = test.merge(df[["x", "y",'trafo']],on=["x", "y"],how='left').set_index(test.index)
        # result['trafo'] = result['trafo'].fillna(False)
        # index = result[result['trafo']==True].index.astype('str').tolist()
        # index2= result.loc[(result['trafo']==True)&(result['country']=='DE'),["y", "x","substation_lv",'substation_off',"v_nom"]].sort_values(by=["x", "y","substation_lv",'substation_off',"v_nom"]).drop_duplicates(["y", "x","v_nom"],keep='first').index.astype('str').tolist()
        # trafo = set(b1.transformers.bus0.astype('str'))|set(b1.transformers.bus1.astype('str'))
        # print(len(set(index)-set(trafo)))
        # ####
        #b1.buses.loc[['190', '32999'],['substation_off','substation_lv','onshore_bus']] = [False, False, False]
        # Löscht AC-Leitung von Rostock nach Dänemark -> existiert nicht
        b1.remove("Bus", ['12263'])
        b1.remove("Line", ['26489','26478'])
        # ergänzt fehlende Leitungen zwischen DE und Nachbarländern
        new_lines = pd.DataFrame(columns=b1.lines.replace('',np.nan).dropna(how='all',axis=1).columns)
        # ergänzt Leitungen zw. DE und NL
        # UW Gronau (Westfalen) -> NL 380kV
        new_lines.loc[str(b0.lines.index.astype('int').max()+1),['bus0', 'bus1','v_nom','num_parallel','length']] = ['28844', '32999', 380.0, 2, 126]
        # UW Niederrhein -> NL 380kV
        new_lines.loc[str(b0.lines.index.astype('int').max()+2),['bus0', 'bus1','v_nom','num_parallel','length']] = ['28344', '32999', 380.0, 2, 144]
        # UW Siersdorf -> NL 380kV
        new_lines.loc[str(b0.lines.index.astype('int').max()+3),['bus0', 'bus1','v_nom','num_parallel','length']] = ['28537', '32999', 380.0, 2, 238]    
        new_lines.loc[:,['active', 'build_year', 'lifetime','carrier','terrain_factor','v_ang_min', 'v_ang_max', 'x_pu',
        'r_pu', 'g_pu', 'b_pu','x_pu_eff', 'r_pu_eff', 's_nom_opt','g']]=[True, 0, 40.0, 'AC', 1.0, -np.inf, np.inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        new_lines["x"] = new_lines["length"] * b1.line_types.at["Al/St 240/40 4-bundle 380.0",'x_per_length']
        new_lines["r"] = new_lines["length"] * b1.line_types.at["Al/St 240/40 4-bundle 380.0",'r_per_length']    
        new_lines["s_nom"] = np.sqrt(3) * b1.line_types.at["Al/St 240/40 4-bundle 380.0",'i_nom'] * new_lines["v_nom"] * new_lines["num_parallel"]
        b1.add("Line", new_lines.index, **new_lines)  

        # def _set_lines_s_nom_from_linetypes(n):
        #     n.lines["s_nom"] = (
        #         np.sqrt(3)
        #         * n.lines["type"].map(n.line_types.i_nom)
        #         * n.lines["v_nom"]
        #         * n.lines["num_parallel"]
        #     )
        # b1.lines[['b', 'capital_cost', 'r', 's_nom_min', 'x','s_nom']] = np.nan
        # b1.lines.loc[b1.lines["v_nom"]==110,"type"] = "243-AL1/39-ST1A 110.0"
        # b1.lines.loc[b1.lines["v_nom"]==220,"type"] = "Al/St 240/40 2-bundle 220.0"
        # b1.lines.loc[b1.lines["v_nom"]==380,"type"] = "Al/St 240/40 4-bundle 380.0"
        # _set_lines_s_nom_from_linetypes(b1)
        
        b1.lines["s_nom_min"] = b1.lines["s_nom"]
        b1.lines["s_max_pu"] = 0.7
        b1.lines["s_nom_extendable"] = False
        b1.links["p_nom_extendable"] = False
        
        if dynamic_line_rating:
            #Regelzonen; WGS84 (EPSG:4326)
            TSO_shp="D:/Ablage/Daten/Stromnetzbetreiber ESRI Shapefile WGS84/2024/Strom-Regelzonen2024_ShapefileWGS84/Strom-Regelzonen2024-01.shp"
            df_TSO= gpd.read_file(TSO_shp)
            df_TSO=df_TSO.set_crs("EPSG:4326")
            df_TSO=df_TSO[df_TSO['Name'].isin(['TransnetBW GmbH','Amprion GmbH','50Hertz Transmission GmbH','TenneT TSO GmbH'])]
        
            b1.lines["underground"] = True
            # b1.lines.loc[b1.lines.v_nom==110,"underground"] = True
            b1.lines["geom"] = b1.lines["topo"]
            b1.lines.loc[~b1.lines['geom'].isna(),'geometry'] = b1.lines.loc[~b1.lines['geom'].isna(),'geom'].apply(wkb.loads, hex=True)
        
            lines_geo = gpd.GeoSeries(b1.lines.loc[~b1.lines['geom'].isna(),'geometry'],crs=4326)
            b1.lines.loc[~b1.lines['geom'].isna(),"Transnet"] = (
                lines_geo.geometry.intersection(df_TSO[df_TSO['Name']=='TransnetBW GmbH'].geometry.iloc[0]).length / lines_geo.geometry.length
            )
            b1.lines.loc[~b1.lines['geom'].isna(),"Amprion"] = (
                lines_geo.geometry.intersection(df_TSO[df_TSO['Name']=='Amprion GmbH'].geometry.iloc[0]).length / lines_geo.geometry.length
            )
            b1.lines.loc[~b1.lines['geom'].isna(),"50Hertz"] = (
                lines_geo.geometry.intersection(df_TSO[df_TSO['Name']=='50Hertz Transmission GmbH'].geometry.iloc[0]).length / lines_geo.geometry.length
            )
            b1.lines.loc[~b1.lines['geom'].isna(),"TenneT"] = (
                lines_geo.geometry.intersection(df_TSO[df_TSO['Name']=='TenneT TSO GmbH'].geometry.iloc[0]).length / lines_geo.geometry.length
            )
            b1.lines[["Transnet", "Amprion", "50Hertz", "TenneT"]] = b1.lines[["Transnet", "Amprion", "50Hertz", "TenneT"]].replace(0,np.nan)
            b1.lines["TSO"] = b1.lines[["Transnet", "Amprion", "50Hertz", "TenneT"]].idxmax(axis=1)
            # Zufällig xx% historischer FLM der Indizes auswählen
            b1_DE_index = b1.buses[b1.buses.country=='DE'].index.tolist()
            for TSO in ["Transnet", "Amprion", "50Hertz", "TenneT"]:
                FLM_380 = b1.lines.loc[(b1.lines["TSO"]==TSO)&(b1.lines.v_nom==380)&(b1.lines.bus0.isin(b1_DE_index))&(b1.lines.bus1.isin(b1_DE_index))]
                if TSO=="50Hertz":
                    num_rows = int(0.64 * len(FLM_380))
                elif TSO=="Amprion":
                    num_rows = int(0.37 * len(FLM_380))
                elif TSO=="TenneT":
                    num_rows = int(0.33 * len(FLM_380))
                elif TSO=="Transnet":
                    num_rows = int(0.86 * len(FLM_380))
                random_indices = np.random.choice(FLM_380.index, size=num_rows, replace=False)
                # Spalte "underground" für diese Zeilen auf False setzen
                b1.lines.loc[random_indices, "underground"] = False
                FLM_220 = b1.lines.loc[(b1.lines["TSO"]==TSO)&(b1.lines.v_nom==220)&(b1.lines.bus0.isin(b1_DE_index))&(b1.lines.bus1.isin(b1_DE_index))]
                if TSO=="50Hertz":
                    num_rows = int(0.0 * len(FLM_220))
                elif TSO=="Amprion":
                    num_rows = int(0.35 * len(FLM_220))
                elif TSO=="TenneT":
                    num_rows = int(0.42 * len(FLM_220))
                elif TSO=="Transnet":
                    num_rows = int(0.74 * len(FLM_220))
                random_indices = np.random.choice(FLM_220.index, size=num_rows, replace=False)
                # Spalte "underground" für diese Zeilen auf False setzen
                b1.lines.loc[random_indices, "underground"] = False
            
            b1.lines.loc[b1.lines.underground==False, 'color'] = 'red'
            lines_geo_FLM = lines_geo.loc[lines_geo.index.isin(b1.lines.loc[b1.lines.underground==False].index)]
            fig, ax = plt.subplots(figsize=(10, 8))
            # fig = plt.figure(figsize=(10, 8))
            lines_geo.loc[lines_geo.index.isin(b1.lines.index)].to_crs("EPSG:4326").plot(ax=ax, color='grey', linewidth=0.8, zorder=1)
            lines_geo_FLM.to_crs("EPSG:4326").plot(ax=ax, color=b1.lines.loc[b1.lines.index.isin(lines_geo_FLM.index),'color'].fillna('grey'), linewidth=0.8, zorder=2)
            df_TSO[df_TSO['Name']=='TenneT TSO GmbH'].to_crs("EPSG:4326").plot(ax=ax,edgecolor='#58508d',color='#58508d',linewidth=0.3,alpha=0.6,label='TenneT',zorder=0)
            df_TSO[df_TSO['Name']=='TransnetBW GmbH'].to_crs("EPSG:4326").plot(ax=ax,edgecolor='#003f5c',color='#003f5c',linewidth=0.3,alpha=0.6,label='TransnetBW',zorder=0)
            df_TSO[df_TSO['Name']=='Amprion GmbH'].to_crs("EPSG:4326").plot(ax=ax,edgecolor='#bc5090',color='#bc5090',linewidth=0.3,alpha=0.6,label='Amprion',zorder=0)
            df_TSO[df_TSO['Name']=='50Hertz Transmission GmbH'].to_crs("EPSG:4326").plot(ax=ax,edgecolor='#ffa600',color='#ffa600',linewidth=0.3,alpha=0.6,label='50Hertz',zorder=0)
            country_shapes.to_crs("EPSG:4326").boundary.plot(ax=ax,edgecolor="black",linewidth=0.3,zorder=-1)
            plt.legend()
            # plt.savefig("C:/pypsa-eur3/resources/"+resources[0]+"/networks/base_line_rating.png", transparent=True, dpi=300,
            #             bbox_inches='tight')
            plt.savefig("C:/" + pypsa_version + "/resources/"+resources[1]+"/networks/base_line_rating.png", transparent=True, dpi=300,
                        bbox_inches='tight')
            plt.close('all')
            
            b1.lines.drop(['geom','topo','geometry','Transnet_share', 'Amprion_share', '50Hertz_share', 'TenneT_share',
            'Transnet', 'Amprion', '50Hertz', 'TenneT', 'TSO','color'], axis=1, errors="ignore", inplace=True)
        else:
            b1.lines.drop(['geom','topo'], axis=1, errors="ignore", inplace=True)
        
        #etrago
        # electrical.py; def delete_ehv_buses_no_lines(network):
        """
        When there are AC buses totally isolated, this function deletes them in
        order to make possible the creation of busmaps based on electrical
        connections and other purposes. Additionally, it throws a warning to
        inform the user in case that any correction should be done.
        """
        lines = b1.lines
        buses_ac = b1.buses[
            (b1.buses.carrier == "AC") #& (b1.buses.country == "DE")
        ]
        buses_in_lines = set(list(lines.bus0) + list(lines.bus1))
        buses_ac["with_line"] = buses_ac.index.isin(buses_in_lines)
        buses_in_links = set(list(b1.links.bus0) + list(b1.links.bus1))
        buses_ac["with_link"] = buses_ac.index.isin(buses_in_links)
        
        delete_buses = buses_ac[
            (~buses_ac["with_line"])
            & (~buses_ac["with_link"])
            & (buses_ac['v_nom']>150)
            ].index
        b1.remove("Bus", delete_buses)
        
        delete_trafo = b1.transformers[
            (b1.transformers.bus0.isin(delete_buses))
            | (b1.transformers.bus1.isin(delete_buses))
            ].index
        b1.remove("Transformer", delete_trafo)
        dangling_generators = b1.generators.loc[~(b1.generators.bus.isin(b1.buses.index))].index.tolist()
        dangling_storage_units = b1.storage_units.loc[~(b1.storage_units.bus.isin(b1.buses.index))].index.tolist()
        dangling_loads = b1.loads.loc[~(b1.loads.bus.isin(b1.buses.index))].index.tolist()
        b1.remove("Generator", dangling_generators)
        b1.remove("StorageUnit", dangling_storage_units)
        b1.remove("Load", dangling_loads)
        
        b1.generators['carrier_orig'] = b1.generators['carrier'].copy()
        b1.generators.loc[b1.generators['carrier_orig'].str.contains('lignite'),'carrier'] = 'lignite'
        b1.generators.loc[b1.generators['carrier_orig'].str.contains('oil'),'carrier'] = 'oil'
        b1.generators.loc[b1.generators['carrier_orig'].str.contains('coal'),'carrier'] = 'coal'
        b1.generators.loc[b1.generators['carrier_orig'].str.contains('biomass'),'carrier'] = 'biomass'
        b1.generators.loc[b1.generators['carrier_orig'].str.contains('others'),'carrier'] = 'others'
        b1.generators.loc[b1.generators['carrier_orig'].str.contains('reservoir'),'carrier'] = 'hydro'
        b1.generators.loc[b1.generators['carrier_orig'].str.contains('run_of_river'),'carrier'] = 'ror'
        b1.generators.loc[b1.generators['carrier_orig'].str.contains('solar'),'carrier'] = 'solar'
        b1.generators.loc[b1.generators['carrier_orig'].str.contains('wind_onshore'),'carrier'] = 'onwind'
        b1.generators.loc[(b1.generators['carrier_orig'].str.contains('wind_offshore'))&(b1.generators['bus'].isin(b1.buses[b1.buses.v_nom<380].index)),'carrier'] = 'offwind-ac'
        b1.generators.loc[(b1.generators['carrier_orig'].str.contains('wind_offshore'))&(b1.generators['bus'].isin(b1.buses[b1.buses.v_nom>=380].index)),'carrier'] = 'offwind-dc'
        b1.storage_units.loc[(b1.storage_units['carrier'].str.contains('pumped_hydro')),'carrier'] = 'PHS'
        

    if not b1.loads.empty:
        b1.remove("Load",b1.loads.index)
    if not b1.storage_units.empty:
        b1.remove("StorageUnit",b1.storage_units.index)
    if not b1.generators.empty:
        b1.remove("Generator",b1.generators.index)
        
    for name in resources:
        if name == "TM-EHV":
            b2 = b1.copy()
            
            def busmap_by_shortest_path(n, fromlvl): # (spatial.py) pypsa-eur simplifiy_network.py #aggregate_to_substations
                """
                Creates a busmap for the EHV-Clustering between voltage levels based
                on dijkstra shortest path.
            
                Parameters
                ----------
                network : pypsa.Network
                    Container for all network components.
                session : sqlalchemy.orm.session.Session object
                    Establishes interactions with the database.
                fromlvl : list
                    List of voltage-levels to cluster.
                tolvl : list
                    List of voltage-levels to remain.
                cpu_cores : int
                    Number of CPU-cores.
            
                Returns
                -------
                None
                """
                # dijkstra's algorithm
                weight = pd.concat(
                    {
                        "Line": n.lines.length.clip(lower=1e-3) , #/ n.lines.s_nom.clip(1e-3)
                        "Link": n.links.length.clip(lower=1e-3) , #/ n.links.p_nom.clip(1e-3)
                        "Transformer": pd.Series(1e-3, index=n.transformers.index),  # Konstantes Gewicht für Transformer
                    }
                )
            
                adj = n.adjacency_matrix(branch_components=["Line", "Link","Transformer"], weights=weight)
            
                # buses_c = list(set(buses_i) - set(medoid_idx))
                bus_indexer = n.buses.index.get_indexer(fromlvl)
                dist = pd.DataFrame(
                    dijkstra(adj, directed=False, indices=bus_indexer), fromlvl, n.buses.index
                )
            
                dc_buses = n.buses[n.buses.carrier=='DC'].index.tolist()
                dist.loc[:, dist.columns.isin(set(fromlvl)|set(dc_buses))] = (
                    np.inf
                )  # bus in buses_i should not be assigned to different bus in buses_i
                # verschieben nur innerhalb eines Landes erlaubt
                for c in n.buses.country.unique():
                    incountry_b = n.buses.country == c
                    dist.loc[incountry_b, ~incountry_b] = np.inf
                busmap = n.buses.index.to_series()
                busmap.loc[fromlvl] = dist.idxmin(1)
            
                return busmap
            
            # spatial.py; def busmap_ehv_clustering
            """
            Generates a busmap that can be used to cluster an electrical network to
            only extra high voltage buses.
            """
            v_nom_min = 150
            v_nom_max = 800
            buses_with_v_nom_to_keep_b = (
                    (v_nom_min <= b2.buses.v_nom) & (b2.buses.v_nom < v_nom_max)
                    | (b2.buses.v_nom.isnull())
                    | (
                            b2.buses.carrier == "DC"
                    )  # Keeping all DC buses from the input dataset independent of voltage (e.g. 150 kV connections)
            )
            ehv_buses = b2.buses[buses_with_v_nom_to_keep_b].index.tolist()
            hv_buses = b2.buses[~buses_with_v_nom_to_keep_b].index.tolist()
            
            busmap = busmap_by_shortest_path(
                b2,
                fromlvl=hv_buses,
                # tolvl=ehv_buses,
                # cpu_cores=cpu_cores,
            )
            pd.DataFrame(busmap.items(), columns=["bus0", "bus1"]).to_csv(
                "C:/" + pypsa_version + "/resources/"+name+"/ehv_elecgrid_busmap_result.csv",
                index=False,
            )
            def _leading(busmap, df): # electrical.py
                """
                Returns a function that computes the leading bus_id for a given mapped
                list of buses.
            
                Parameters
                -----------
                busmap : dict
                    A dictionary that maps old bus_ids to new bus_ids.
                df : pandas.DataFrame
                    A DataFrame containing network.buses data. Each row corresponds
                    to a unique bus
            
                Returns
                --------
                leader : function
                    A function that returns the leading bus_id for the argument `x`.
                """
            
                def leader(x):
                    ix = busmap[x.index[0]]
                    return df.loc[ix, x.name]
            
                return leader
            
            def cluster_on_extra_high_voltage(n, busmap): # electrical.py
                """
                Main function of the EHV-Clustering approach. Creates a new clustered
                pypsa.Network given a busmap mapping all bus_ids to other bus_ids of the
                same network.
            
                Parameters
                ----------
                etrago : Etrago
                    An instance of the Etrago class
                busmap : dict
                    Maps old bus_ids to new bus_ids.
                with_time : bool
                    If true time-varying data will also be aggregated.
            
                Returns
                -------
                network : pypsa.Network
                    Container for all network components of the clustered network.
                busmap : dict
                    Maps old bus_ids to new bus_ids including all sectors.
                """
           
                buses = aggregatebuses(
                    n,
                    busmap,
                    {
                        "x": _leading(busmap, n.buses),
                        "y": _leading(busmap, n.buses),
                        "substation_lv": lambda x: bool(x.sum()),
                        "substation_off": lambda x: bool(x.sum()),
                        "onshore_bus": lambda x: bool(x.sum()),
                    },
                )
                #drop_buses = n.buses.loc[~((n.buses.index.isin(buses.index)))].index.tolist()
                drop_lines = n.lines.loc[~((n.lines.bus0.isin(buses.index)) & (n.lines.bus1.isin(buses.index)))].index.tolist()
                drop_transformers = n.transformers.loc[~
                    ((n.transformers.bus0.isin(buses.index)) & (n.transformers.bus1.isin(buses.index)))].index.tolist()
                ehv = n.copy()
                ehv.remove("Bus", ehv.buses.index)
                ehv.add("Bus", name=buses.index,  **buses)
                ehv.remove("Line", drop_lines)
                ehv.remove("Transformer", drop_transformers)
                # Dealing with links
                # links = n.links.copy()
                # dc_links = links[links["carrier"] == "DC"]
                # # Discard links connected to buses under 220 kV
                # dc_links = dc_links[dc_links.bus0.isin(buses.index)]
                # links = links[links["carrier"] != "DC"]
            
                new_links = (
                    n.links.assign(bus0=n.links.bus0.map(busmap), bus1=n.links.bus1.map(busmap))
                    .dropna(subset=["bus0", "bus1"])
                    .loc[lambda df: df.bus0 != df.bus1]
                )
            
                ehv.remove("Link", ehv.links.index)
                ehv.add("Link", name=new_links.index,  **new_links)
            
                return ehv
            
            ehv = cluster_on_extra_high_voltage(
                b2, busmap
            )
            
            n = ehv.copy()
            
            ehv.lines.loc[ehv.lines.underground==False, 'color'] = 'red'
            lines_geo_FLM = lines_geo.loc[lines_geo.index.isin(ehv.lines.loc[ehv.lines.underground==False].index)]
            fig, ax = plt.subplots(figsize=(10, 8))
            # fig = plt.figure(figsize=(10, 8))
            lines_geo.loc[lines_geo.index.isin(ehv.lines.index)].to_crs("EPSG:4326").plot(ax=ax, color='grey', linewidth=0.8, zorder=1)
            lines_geo_FLM.to_crs("EPSG:4326").plot(ax=ax, color=ehv.lines.loc[ehv.lines.index.isin(lines_geo_FLM.index),'color'].fillna('grey'), linewidth=0.8, zorder=2)
            df_TSO[df_TSO['Name']=='TenneT TSO GmbH'].to_crs("EPSG:4326").plot(ax=ax,edgecolor='#58508d',color='#58508d',linewidth=0.3,alpha=0.6,label='TenneT',zorder=0)
            df_TSO[df_TSO['Name']=='TransnetBW GmbH'].to_crs("EPSG:4326").plot(ax=ax,edgecolor='#003f5c',color='#003f5c',linewidth=0.3,alpha=0.6,label='TransnetBW',zorder=0)
            df_TSO[df_TSO['Name']=='Amprion GmbH'].to_crs("EPSG:4326").plot(ax=ax,edgecolor='#bc5090',color='#bc5090',linewidth=0.3,alpha=0.6,label='Amprion',zorder=0)
            df_TSO[df_TSO['Name']=='50Hertz Transmission GmbH'].to_crs("EPSG:4326").plot(ax=ax,edgecolor='#ffa600',color='#ffa600',linewidth=0.3,alpha=0.6,label='50Hertz',zorder=0)
            country_shapes.to_crs("EPSG:4326").boundary.plot(ax=ax,edgecolor="black",linewidth=0.3,zorder=-1)
            plt.legend()
            plt.savefig("C:/" + pypsa_version + "/resources/"+resources[0]+"/networks/base_line_rating.png", transparent=True, dpi=300,
                        bbox_inches='tight')
            plt.close('all')
        
        else:
            n = b1.copy()
            # b2 = b1.copy()
            
        graph = n.graph()
        is_connected = nx.is_connected(graph)
        print(f"Das Netzwerk ist zusammenhängend: {is_connected}")
        if is_connected == False:
            components = list(nx.connected_components(graph))
            print(f"Anzahl der zusammenhängenden Komponenten: {len(components)}")
            for i, component in enumerate(components, start=1):
                print(f"Komponente {i}: {component}")
                
        if 'eTraGo' not in folder_name:
            def _set_links_underwater_fraction(n, offshore_shapes):
                if n.links.empty:
                    return
            
                if not hasattr(n.links, "geometry"):
                    n.links["underwater_fraction"] = 0.0
                else:
                    offshore_shape = gpd.read_file(offshore_shapes).union_all()
                    links = gpd.GeoSeries(n.links.geometry.dropna().map(shapely.wkt.loads))
                    n.links["underwater_fraction"] = (
                        links.intersection(offshore_shape).length / links.length
                    )
            
            _set_links_underwater_fraction(n, offshore_shape)
        
        def _set_shapes(n, country_shapes, offshore_shapes):
            # Write the geodataframes country_shapes and offshore_shapes to the network.shapes component
            country_shapes = gpd.read_file(country_shapes).rename(columns={"name": "idx"})
            country_shapes["type"] = "country"
            offshore_shapes = gpd.read_file(offshore_shapes).rename(columns={"name": "idx"})
            offshore_shapes["type"] = "offshore"
            all_shapes = pd.concat([country_shapes, offshore_shapes], ignore_index=True)
            n.add(
                "Shape",
                all_shapes.index,
                geometry=all_shapes.geometry,
                idx=all_shapes.idx,
                type=all_shapes["type"],
            )
        
        _set_shapes(n, country_shape, offshore_shape)
        
        # Add carriers if they are present in buses.carriers
        carriers_in_buses = set(n.buses.carrier.dropna().unique())
        carriers = carriers_in_buses.intersection({"AC", "DC"})
        
        if carriers:
            n.add("Carrier", carriers)
                

        if "geometry" in n.links:
            # Konvertiere Geometrien in WKT-Format
            n.links["geometry"] = n.links["geometry"].apply(
                lambda geom: geom.wkt if geom else None
            )

        # n.lines.loc[n.lines.v_nom==380.0, 'color'] = 'red'
        # n.lines.loc[n.lines.v_nom==220.0, 'color'] = 'green'
        # n.lines.loc[n.lines.v_nom==110.0, 'color'] = 'grey'
        # fig = plt.figure(figsize=(10, 8))
        # n.plot(line_colors=n.lines.color.fillna('grey'), link_colors='orange', bus_sizes=0, bus_alpha=0,
                 # line_widths=0.8, link_widths=0.8)
        # plt.savefig("C:/" + pypsa_version + "/resources/"+name+"/networks/base_simpl_voltagelevel.png", transparent=True, dpi=300,
                    # bbox_inches='tight')
        # plt.close('all')
        
        # n.lines.drop(['color'], axis=1, errors="ignore", inplace=True)
        
        # n.export_to_netcdf("C:/" + pypsa_version + "/resources/"+name+"/networks/base.nc")
        
    return n
