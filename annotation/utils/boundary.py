import ee
from google.api_core import retry as retry_google
import google
import numpy as np
import time

DATE_CHANGE = '2023-12-31'

@retry_google.Retry()
def get_working_boundary_offset(center_pos, scale_x, scale_y, patch_size, osm_format=True):
    """
    Get the boundary by calculate the offset between the center to the working boundary.
    
    Args:
        center_pos (tuple): Center position of the boundary (lon, lat).
        scale_x (float): Unit variance for a pixel in x axis
        scale_y (float): Unit variance for a pixel in y axis
        patch_size (int): Patch size in pixels.
        osm_format (bool, optional): Whether to return the boundary in OSM format. Defaults to True.
                                    osm format:  [min_lat, min_lon, max_lat, max_lon]
                                    ee format:  [min_lon, min_lat, max_lon, max_lat]
    
    Returns:
        tuple: Boundary either in osm format or ee format.
    """
    
    offset_x = abs(scale_x * patch_size / 2)
    offset_y = abs(scale_y * patch_size / 2)
    
    x_min, x_max = center_pos[0] - offset_x, center_pos[0] + offset_x
    y_min, y_max = center_pos[1] - offset_y, center_pos[1] + offset_y
    
    if osm_format:
        return [y_min, x_min, y_max, x_max]
    else:
        return [x_min, y_min, x_max, y_max]
    
@retry_google.Retry()    
def get_working_boundary_buffer(center_pos, scale, patch_size, osm_format=True):
    """
    Get the boundary by using the ee.Geometry.buffer method

    Args:
        center_pos (tuple): Center position of the boundary (lon, lat).
        scale (float): Pixel resolusion in meters
        patch_size (int): Patch size in pixels
        osm_format (bool, optional): Whether to return the boundary in OSM format. Defaults to True.
                                    osm format:  [min_lat, min_lon, max_lat, max_lon]
                                    ee format:  [min_lon, min_lat, max_lon, max_lat]

    Returns:
        tuple: Boundary either in osm format or ee format.
    """
    point = ee.Geometry.Point(center_pos)
    bounds_ee = point.buffer(scale * patch_size / 2).bounds()
    bounds = bounds_ee.getInfo()
    bounds_np = np.array(bounds['coordinates'][0])
    x_min, x_max, y_min, y_max = bounds_np[:,0].min(), bounds_np[:,0].max(),\
                                 bounds_np[:,1].min(), bounds_np[:,1].max()
                                 
    if osm_format:
        return [y_min, x_min, y_max, x_max]
    else:
        return [x_min, y_min, x_max, y_max]
    
def get_osm_working_boundary(center_pos, patch_dimension, download_time, patch_crs, scale):
    """
    Choose the boundary calculation function between `get_working_boundary_offset` and `get_working_boundary_buffer`.
    For the data downloaded before `2023-12-05` we use `get_working_boundary_offset`.

    Args:
        center_pos (tuple): (lat, lon) of the center of the patch
        patch_dimension (int): dimension of the patch in pixels
        download_time (str): timestamp of the data downloaded
        patch_crs (str): coordinate reference system of the patch
        scale (float): scale of the patch in meters per pixel

    Returns:
        tuple: (min_lat, min_lon, max_lat, max_lon) of the working boundary
    """

    download_time_ticks = time.mktime(time.strptime(download_time, '%Y-%m-%d %H:%M:%S'))
    change_time_ticks = time.mktime(time.strptime(DATE_CHANGE, '%Y-%m-%d'))
    
    function_index = 0 if download_time_ticks < change_time_ticks else 1
    
    if function_index == 0:
        # Make a projection to discover the scale in degrees.
        patch_proj = ee.Projection(patch_crs).atScale(scale).getInfo()

        # Get scales in degrees out of the transform.
        scale_x = patch_proj['transform'][0]
        scale_y = patch_proj['transform'][4]
        boundary = get_working_boundary_offset(center_pos, scale_x, scale_y, patch_dimension)
        
    else:
        boundary = get_working_boundary_buffer(center_pos, scale, patch_dimension)
        
    return boundary