from typing import Dict
from tqdm import tqdm
from shapely.geometry import shape, LinearRing, MultiLineString
from shapely import from_wkt
import numpy as np
import concurrent.futures
import urllib
from retry import retry
import sqlite3
import OSMPythonTools



WAY_AREA_RULES=[
    {
        "key": "area",
        "polygon": "all"
    },
    {
        "key": "building",
        "polygon": "all"
    },
    {
        "key": "highway",
        "polygon": "whitelist",
        "values": [
            "services",
            "rest_area",
            "escape",
            "elevator"
        ]
    },
    {
        "key": "natural",
        "polygon": "blacklist",
        "values": [
            "coastline",
            "cliff",
            "ridge",
            "arete",
            "tree_row"
        ]
    },
    {
        "key": "landuse",
        "polygon": "all"
    },
    {
        "key": "waterway",
        "polygon": "whitelist",
        "values": [
            "riverbank",
            "dock",
            "boatyard",
            "dam"
        ]
    },
    {
        "key": "amenity",
        "polygon": "all"
    },
    {
        "key": "leisure",
        "polygon": "all"
    },
    {
        "key": "barrier",
        "polygon": "whitelist",
        "values": [
            "city_wall",
            "ditch",
            "hedge",
            "retaining_wall",
            "wall",
            "spikes"
        ]
    },
    {
        "key": "railway",
        "polygon": "whitelist",
        "values": [
            "station",
            "turntable",
            "roundhouse",
            "platform"
        ]
    },
    {
        "key": "boundary",
        "polygon": "all"
    },
    {
        "key": "man_made",
        "polygon": "blacklist",
        "values": [
            "cutline",
            "embankment",
            "pipeline"
        ]
    },
    {
        "key": "power",
        "polygon": "whitelist",
        "values": [
            "plant",
            "substation",
            "generator",
            "transformer"
        ]
    },
    {
        "key": "place",
        "polygon": "all"
    },
    {
        "key": "shop",
        "polygon": "all"
    },
    {
        "key": "aeroway",
        "polygon": "blacklist",
        "values": [
            "taxiway"
        ]
    },
    {
        "key": "tourism",
        "polygon": "all"
    },
    {
        "key": "historic",
        "polygon": "all"
    },
    {
        "key": "public_transport",
        "polygon": "all"
    },
    {
        "key": "office",
        "polygon": "all"
    },
    {
        "key": "building:part",
        "polygon": "all"
    },
    {
        "key": "military",
        "polygon": "all"
    },
    {
        "key": "ruins",
        "polygon": "all"
    },
    {
        "key": "area:highway",
        "polygon": "all"
    },
    {
        "key": "craft",
        "polygon": "all"
    },
    {
        "key": "golf",
        "polygon": "all"
    },
    {
        "key": "indoor",
        "polygon": "all"
    }
]


def is_area_from_way_tags(tags: Dict):
    """
    A way is considered a Polygon if

        It forms a closed loop, and
        it is not tagged area=no, and
        at least one of the following conditions is true:
        there is a area=* tag;
        there is a area:highway=* tag and its value is not: no;
        there is a aeroway=* tag and its value is not any of: no, or taxiway;
        there is a amenity=* tag and its value is not: no;
        there is a barrier=* tag and its value is one of: city_wall, ditch, hedge, retaining_wall, wall or spikes;
        there is a boundary=* tag and its value is not: no;
        there is a building:part=* tag and its value is not: no;
        there is a building=* tag and its value is not: no;
        there is a craft=* tag and its value is not: no;
        there is a golf=* tag and its value is not: no;
        there is a highway=* tag and its value is one of: services, rest_area, escape or elevator;
        there is a historic=* tag and its value is not: no;
        there is a indoor=* tag and its value is not: no;
        there is a landuse=* tag and its value is not: no;
        there is a leisure=* tag and its value is not: no;
        there is a man_made=* tag and its value is not any of: no, cutline, embankment nor pipeline;
        there is a military=* tag and its value is not: no;
        there is a natural=* tag and its value is not any of: no, coastline, cliff, ridge, arete nor tree_row;
        there is a office=* tag and its value is not: no;
        there is a place=* tag and its value is not: no;
        there is a power=* tag and its value is one of: plant, substation, generator or transformer;
        there is a public_transport=* tag and its value is not: no;
        there is a railway=* tag and its value is one of: station, turntable, roundhouse or platform;
        there is a ruins=* tag and its value is not: no;
        there is a shop=* tag and its value is not: no;
        there is a tourism=* tag and its value is not: no;
        there is a waterway=* tag and its value is one of: riverbank, dock, boatyard or dam;
    Args:
        tags (dict): _description_
    """
    
    is_area = False

    for rule in WAY_AREA_RULES:
        key = rule['key']
        policy = rule['polygon']
        value = tags.get(key, None)
        
        if value == None:
            continue
        
        if policy == 'all':
            if value != 'no':
                is_area = True
                break
        elif policy == 'blacklist':
            policy_value = rule['values']
            if value not in policy_value:
                is_area = True
                break
        else: # policy == 'whitelist'
            policy_value = rule['values']
            if value in policy_value:
                is_area = True
                break            
    
    return is_area

def is_area_from_relation_tags(tags: Dict, name_required=False):
    """Following the official implementation of osm
    
        <?xml version="1.0" encoding="UTF-8"?>
        <osm-script timeout="86400" element-limit="4294967296">

        <union>
        <query type="relation">
            <has-kv k="type" v="multipolygon"/>
            <has-kv k="name"/>
        </query>
        <query type="relation">
            <has-kv k="type" v="boundary"/>
            <has-kv k="name"/>
        </query>
        <query type="relation">
            <has-kv k="admin_level"/>
            <has-kv k="name"/>
        </query>
        <query type="relation">
            <has-kv k="postal_code"/>
        </query>
        <query type="relation">
            <has-kv k="addr:postcode"/>
        </query>
        </union>
        <foreach into="pivot">
        <union>
            <recurse type="relation-way" from="pivot"/>
            <recurse type="way-node"/>
        </union>
        <make-area pivot="pivot" return-area="no"/>
        </foreach>

        </osm-script>   

    Args:
        tags (Dict): _description_

    Returns:
        _type_: _description_
    """
    
    r_type = tags.get('type', False)
    name =  tags.get('name', False)
    name = name or not name_required
    
    if r_type in ('multipolygon', 'boundary') and name:
        return True
        
    admin_level = tags.get('admin_level',None)
    if admin_level and name:
        return True
        
    postal_code = tags.get('postal_code',None)
    if postal_code:
        return True
    
    addr_postcode = tags.get('addr:postcode',None)
    if addr_postcode:
        return True
    
    return False


def get_way_infos(response):
    
    way_infos = []
    
    for way in tqdm(response.ways()):
         
        way_info = dict()
        out_boundary = False
        tags = way.tags()    
        if tags == None:
            continue    
        
        if len(tags)==1 and 'created_by' in set(tags.keys()):
            continue
        
        is_area = is_area_from_way_tags(tags)
        
        geo = way.geometry()
            
        # We use the intersection between the way geometry and the working area
        shape_geo = shape(geo)    
        
        way_info['geometry'] = shape_geo
        
        way_info['id'] = way.id()
        
        way_info.update(tags = dict(tags) if tags else {})    
        
        if is_area:
            way_infos.append(way_info)
            
    return way_infos

def get_way_info_single(way):
    
    way_info = dict()
    tags = way.tags()    
    
    if tags == None:
        return None
    
    if len(tags)==1 and 'created_by' in set(tags.keys()):
        return None
    
    is_area = is_area_from_way_tags(tags)
    
    if not is_area:
        return None
    
    geo = way.geometry()
    
    # We use the intersection between the way geometry and the working area
    shape_geo = shape(geo)    
    
    way_info['geometry'] = shape_geo
    
    way_info['id'] = way.id()
    
    way_info.update(tags = dict(tags) if tags else {})    
    
    return way_info

def get_way_infos_parallel(response, workers=4):

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(get_way_info_single, way) for way in response.ways()]
        results = [future.result() for future in futures]
        
    return [item for item in results if item is not None]

def get_relation_infos(response):

    relation_infos = []

    for relation in tqdm(response.relations()):

        relation_info = dict()
        tags = relation.tags() 
        
        if tags == None:
            continue    
        
        if len(tags)==1 and 'created_by' in set(tags.keys()):
            continue
        
        is_area = is_area_from_relation_tags(tags)
        
        if not is_area:
            continue
        
        try:
            geo = relation.geometry()
        except Exception as exc:
            print(exc)
            continue
            
        # We use the intersection between the way geometry and the working area
        shape_geo = shape(geo)    
        
        relation_info['geometry'] = shape_geo
        
        relation_info['id'] = relation.id()
        
        relation_info.update(tags = dict(tags) if tags else {})    
        
        relation_infos.append(relation_info)
        
    return relation_infos

def get_relation_info_single(relation):
    
    relation_info = dict()
    tags = relation.tags() 

    if tags == None:
        return None   

    if len(tags)==1 and 'created_by' in set(tags.keys()):
        return None   

    is_area = is_area_from_relation_tags(tags)

    if not is_area:
        return None   

    try:
        geo = relation.geometry()
    except Exception as exc:
        print(exc)
        return None   
        
    # We use the intersection between the way geometry and the working area
    shape_geo = shape(geo)    

    relation_info['geometry'] = shape_geo

    relation_info['id'] = relation.id()

    relation_info.update(tags = dict(tags) if tags else {})    

    return relation_info
    
def get_relation_infos_parallel(response, workers=4):

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(get_relation_info_single, relation) for relation in response.relations()]
        results = [future.result() for future in futures]
        
    return [item for item in results if item is not None]


def get_area_nonearea_from_response(way_response, relation_response):

    osm_elements=dict(area=[], nonearea=[])

    for way in way_response.elements():
        
        way_info = dict()
        tags = way.tags()    
        
        if tags == None:
            continue    
        
        if len(tags)==1 and 'created_by' in set(tags.keys()):
            continue

        way_type = 'area' if is_area_from_way_tags(tags) else 'nonearea'
        
        geo = way.geometry()
            
        # We use the intersection between the way geometry and the working area
        shape_geo = shape(geo)    
        
        if shape_geo.geom_type == 'Polygon' and way_type == 'nonearea':
            if len(geo['coordinates'])==1:
                shape_geo = LinearRing(geo['coordinates'][0])
            else:
                shape_geo = MultiLineString(geo['coordinates'])
        
        way_info['geometry'] = shape_geo
        
        way_info['id'] = way.id()
        
        way_info.update(tags = dict(tags) if tags else {})    
        
        osm_elements[way_type].append(way_info)
        
    # iterate through relation data and collect those are areas and their geometries
    for relation in relation_response.elements():

        relation_info = dict()
        tags = relation.tags() 
        
        if len(tags)==1 and 'created_by' in set(tags.keys()):
            continue
        
        relation_type = 'area' if is_area_from_relation_tags(tags) else 'nonarea'
        
        try:
            geo = relation.geometry()
        except Exception as exc:
            # print(exc)
            continue
            
        # We use the intersection between the way geometry and the working area
        shape_geo = shape(geo)    
        
        relation_info['geometry'] = shape_geo
        
        relation_info['id'] = relation.id()
        
        relation_info.update(tags = dict(tags) if tags else {})    
        
        osm_elements[relation_type].append(relation_info)
        
    return osm_elements

def get_way_info_single2(way):
    
    way_info = dict()
    tags = way.tags()    
    
    if tags == None:
        return None    
    
    if len(tags)==1 and 'created_by' in set(tags.keys()):
        return None 

    way_type = 'area' if is_area_from_way_tags(tags) else 'nonearea'
    
    try:
        geo = way.geometry()
    except:
        return None
        
    # We use the intersection between the way geometry and the working area
    shape_geo = shape(geo)    
    
    if shape_geo.geom_type == 'Polygon' and way_type == 'nonearea':
        if len(geo['coordinates'])==1:
            shape_geo = LinearRing(geo['coordinates'][0])
        else:
            shape_geo = MultiLineString(geo['coordinates'])
    
    way_info['geometry'] = shape_geo
    
    way_info['id'] = way.id()
    
    way_info.update(tags = dict(tags) if tags else {})    
    
    return way_type, way_info  

@retry(urllib.request.HTTPError, tries=3, delay=1, backoff=2)
def get_relation_info_with_db(relation, conn, time_stamp):
    
    relation_info = dict()
    tags = relation.tags() 
    
    if len(tags)==1 and 'created_by' in set(tags.keys()):
        return None
    
    if 'admin_level' in set(tags.keys()):
        return None
    
    if 'boundary' in set(tags.keys()):
        return None
    
    relation_info['id'] = relation.id()
    
    relation_type = 'area' if is_area_from_relation_tags(tags) else 'nonearea'
    
    c = conn.cursor()
    
    c.execute("SELECT GEOMETRY FROM osm_data WHERE OSM_ID =? AND DATA_DATE =?", (relation.id(), time_stamp))
    
    row = c.fetchone()
    
    if row is not None:
        geo = from_wkt(row[0])
        OSMPythonTools.logger.info('Geometry found in database')
    else:
        try:
            geo = relation.geometry()
        except urllib.request.HTTPError as err:
            raise err
        except Exception as exc:
            # print(exc)
            return None 
        
    # We use the intersection between the way geometry and the working area
    shape_geo = shape(geo)    
    
    relation_info['geometry'] = shape_geo
    
    relation_info.update(tags = dict(tags) if tags else {})    
    
    return relation_type, relation_info      
    

@retry(urllib.request.HTTPError, tries=3, delay=1, backoff=2)
def get_relation_info_single2(relation):

    relation_info = dict()
    tags = relation.tags() 
    
    if len(tags)==1 and 'created_by' in set(tags.keys()):
        return None
    
    if 'admin_level' in set(tags.keys()):
        return None
    
    if 'boundary' in set(tags.keys()):
        return None
    
    relation_type = 'area' if is_area_from_relation_tags(tags) else 'nonearea'
    
    try:
        geo = relation.geometry()
    except urllib.request.HTTPError as err:
        raise err
    except Exception as exc:
        # print(exc)
        return None 
        
    # We use the intersection between the way geometry and the working area
    shape_geo = shape(geo)    
    
    relation_info['geometry'] = shape_geo
    
    relation_info['id'] = relation.id()
    
    relation_info.update(tags = dict(tags) if tags else {})    
    
    return relation_type, relation_info  

def get_area_nonearea_from_response_with_db(ways, relations, db_path, time_stamp):
    
    conn = sqlite3.connect(db_path, timeout=120)
    
    data_date = time_stamp.strftime("%Y-%m-%d")
    
    osm_elements=dict(area=[], nonearea=[])

    if ways is not None:
        for way in ways:
            result= get_way_info_single2(way)
            if result is not None:
                osm_elements[result[0]].append(result[1])
                
    # we only query database for histroy relations
    if relations is not None:
        for relation in relations:
            result = get_relation_info_with_db(relation, conn, data_date)
            if result is not None:
                relation_type, relation_info = result
                osm_elements[relation_type].append(relation_info)
                
    conn.close()
    
    return osm_elements

def get_area_nonearea_from_response_parallel(ways, relations, workers=4):
    
    osm_elements=dict(area=[], nonearea=[])
    
    if ways is not None:
        for way in ways:
            result= get_way_info_single2(way)
            if result is not None:
                osm_elements[result[0]].append(result[1])
                
    if relations is not None:
        for relation in relations:
            result = get_relation_info_single2(relation)
            if result is not None:
                osm_elements[result[0]].append(result[1])
                
    return osm_elements

