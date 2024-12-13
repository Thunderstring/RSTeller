import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from random import random
from libpysal.weights import KNN
from shapely.geometry import box, Point, LineString, Polygon
from shapely.affinity import affine_transform
import logging
import contextily
import utm
import json
import sqlite3
import cv2
from .boundary import get_osm_working_boundary

from shapely import from_wkt
from shapely.geometry import box


def get_line_coord_str(geom, decimals=-1):

    out_str = "("

    geom_type = geom.geom_type

    if geom_type == "LineString":
        geom = [geom]
    else:
        geom = list(geom.geoms)

    for g in geom:
        coords = list(g.coords)
        out_str += "("
        vertex = []
        for coord in coords:
            x_pos, y_pos = coord
            vertex.append(
                "({x_pos:.{prec}f}, {y_pos:.{prec}f})".format(
                    x_pos=x_pos, y_pos=y_pos, prec=decimals
                )
            )
        out_str += ", ".join(vertex)
        out_str += "),"
    out_str = out_str[:-1]
    out_str += ")"

    return out_str


def get_line_coord_chess_grid(geom, converter, ignore_same_grid=True):

    out_str = "("

    geom_type = geom.geom_type

    if geom_type == "LineString":
        geom = [geom]
    else:
        geom = list(geom.geoms)

    for g in geom:
        coords = list(g.coords)
        out_str += "("
        vertex = ["NULL"]
        for coord in coords:
            x_pos, y_pos = coord
            grid_label = converter((x_pos, y_pos))
            if not ignore_same_grid or grid_label != vertex[-1]:
                vertex.append(grid_label)
        out_str += ", ".join(vertex[1:])
        out_str += "), "
    out_str = out_str[:-2]
    out_str += ")"

    return out_str


def clip_gdf_by_bbox(gdf, bbox, indicator=True):
    """_summary_

    Args:
        gdf (geoDataFrame): The input geoDataFrame.
        bbox (List[float]): bbox of the region of interest. Format: (xmin, ymin, xmax, ymax)
        indicator (bool, optional): Whether to add a 'clipped' column to the output geoDataFrame indicating whether each geometry is clipped or not. Defaults to True.

    Returns:
        geoDataFrame: The clipped geoDataFrame.
    """
    masked_gdf = gdf.clip(box(*bbox))
    if indicator:
        masked_gdf["clipped"] = gdf.geometry.within(box(*bbox)).apply(lambda x: not x)
    return masked_gdf


class GeometryNormalizer(object):

    def __init__(
        self,
        roi_bbox: list,
        area_norm_tasks=["area", "geometry", "center"],
        nonearea_norm_tasks=["length", "geometry"],
        data_crs="WGS84",
        proj_epsg="UTM",
    ):
        """
        # TODO: the current implementation for calculating length and area is not accurate, need to be improved.
        The inaccuracies are mainly caused by the fact that the length and area are calculated based on the projection of the data, which may not be accurate enough.
        Further studies on find a good projection for length and area calculation is needed.
        Args:
            roi_bbox (list): The ROI given in the form of bbox (xmin, ymin, xmax, ymax).
            area_norm_tasks (list, optional):. Defaults to ['area', 'geometry', 'center'].
            nonearea_norm_tasks (list, optional): Defaults to ['length', 'geometry'].
            data_crs (str, optional): Defaults to 'WGS84'.
            proj_crs (str, optional): Defaults to 'UTM', in this setting, epsg will be calculated based on the center of the ROI.
        """
        self.area_norm_tasks = area_norm_tasks
        self.nonearea_norm_tasks = nonearea_norm_tasks
        self.data_crs = data_crs
        self.roi_bbox = roi_bbox
        self.roi_center = (
            np.mean([roi_bbox[0], roi_bbox[2]]),
            np.mean([roi_bbox[1], roi_bbox[3]]),
        )
        if proj_epsg == "UTM":
            # get epsg code of the UTM zone
            _utm = utm.from_latlon(*self.roi_center)
            self.proj_epsg = (
                32600 + _utm[2] if self.roi_center[1] >= 0 else 32700 + _utm[2]
            )
            # get the scaler of the current UTM zone
            self.scaler = 0.9996
        else:
            self.proj_epsg = proj_epsg
            # get the scaler of the current projection
            if self.proj_epsg == 2062:
                self.scaler = 0.998809
            else:
                raise "Not implemented yet"

        self.roi_gs = gpd.GeoSeries(box(*self.roi_bbox), crs=self.data_crs)
        # get the affine matrix for coordinate normalization
        xmin, ymin, xmax, ymax = self.roi_gs.boundary.bounds.iloc[0]
        a, b, d, e, xoff, yoff = (
            1 / (xmax - xmin),
            0,
            0,
            1 / (ymax - ymin),
            xmin / (xmin - xmax),
            ymin / (ymin - ymax),
        )
        self.norm_affine_matrix = [a, b, d, e, xoff, yoff]

        self.roi_gs = self.roi_gs.to_crs(epsg=self.proj_epsg)
        self.roi_area = self.roi_gs[0].area / (self.scaler**2)

    def _normalize_area(self, gpd):

        gpd = gpd.to_crs(epsg=self.proj_epsg)
        gpd["area"] = gpd.geometry.area / (self.scaler**2)
        gpd["normed_area"] = gpd["area"] / self.roi_area
        gpd = gpd.to_crs(self.data_crs)
        return gpd

    def _normalize_center(self, gpd):

        gpd["center"] = gpd["geometry"].apply(lambda x: x.centroid)
        gpd["normed_center"] = gpd["center"].apply(
            lambda geom: affine_transform(geom, self.norm_affine_matrix)
        )
        return gpd

    def _normalize_geometry(self, gpd):

        gpd["normed_geometry"] = gpd.geometry.apply(
            lambda geom: affine_transform(geom, self.norm_affine_matrix)
        )
        return gpd

    def _normalize_length(self, gpd):

        gpd = gpd.to_crs(epsg=self.proj_epsg)
        gpd["length"] = gpd.length / self.scaler
        gpd["normed_length"] = gpd["length"] / np.sqrt(self.roi_area)
        gpd = gpd.to_crs(self.data_crs)
        return gpd

    def area_normalization(self, gpd):

        for task in self.area_norm_tasks:
            func = getattr(self, "_normalize_{}".format(task))
            gpd = func(gpd)
        return gpd

    def nonearea_normalization(self, gpd):

        for task in self.nonearea_norm_tasks:
            func = getattr(self, "_normalize_{}".format(task))
            gpd = func(gpd)
        return gpd


class Coord2ChessGridMapper(object):

    def __init__(self, chess_grid_size: int, edge_indicator=True):

        assert chess_grid_size > 0, "Chess grid size must be greater than 0"
        assert chess_grid_size < 27

        self.chess_grid_size = chess_grid_size
        self.edge_indicator = edge_indicator

        self.x_grids = ["{}".format(i + 1) for i in range(self.chess_grid_size)]
        self.y_grids = ["{}".format(chr(65 + i)) for i in range(self.chess_grid_size)]

        # an extra grid in case x or y equals 1
        self.x_grids.append("{}".format(self.chess_grid_size))
        self.y_grids.append("{}".format(chr(64 + self.chess_grid_size)))

        self.x_edges = ["_left", "_right"] if self.edge_indicator else ["", ""]
        self.y_edges = ["_bottom", "_top"] if self.edge_indicator else ["", ""]

    def convert(self, coordx: tuple):

        x, y = coordx

        x_grid = self.x_grids[int(x * self.chess_grid_size)]
        y_grid = self.y_grids[int(y * self.chess_grid_size)]
        grid_label = x_grid + y_grid

        if not self.edge_indicator:
            return grid_label

        x_edge = [np.isclose(x, 0), np.isclose(x, 1)]
        y_edge = [np.isclose(y, 0), np.isclose(y, 1)]

        if True in x_edge:
            grid_label += self.x_edges[x_edge.index(True)]
        if True in y_edge:
            grid_label += self.y_edges[y_edge.index(True)]
        return grid_label


class Coord2NineGridMapper(Coord2ChessGridMapper):

    def __init__(self, edge_indicator=True):

        super().__init__(3, edge_indicator=edge_indicator)
        self.edge_indicator = edge_indicator

        self.x_grids = ["left,", "center,", "right,"]
        self.y_grids = ["bottom,", "center,", "top,"]

        # an extra grid in case x or y equals 1
        self.x_grids.append(self.x_grids[-1])
        self.y_grids.append(self.y_grids[-1])

        self.x_edges = ["_left,", "_right,"] if self.edge_indicator else ["", ""]
        self.y_edges = ["_bottom,", "_top,"] if self.edge_indicator else ["", ""]

    def convert(self, coordx: tuple):
        raw = super().convert(coordx)[:-1]

        raw_corse_pos = raw.split(",")
        corse_pos = raw_corse_pos[:2]
        edge_indicator = ""

        if len(raw_corse_pos) == 3:
            edge_indicator = raw_corse_pos[2]

        if corse_pos[0] == "center":
            corse_pos.reverse()
        corse_pos = "-".join(corse_pos)
        if corse_pos == "center-center":
            corse_pos = "center"

        corse_pos += edge_indicator

        return corse_pos


class BaseInterpreter(object):

    _TAG_TEMPLATE_K = """Its key is {key}, which means {description}. The tag belongs to a tag group '{tgroup}'. The tag value is {value}."""
    _TAG_TEMPLATE_KV = """{key}:{value}. The tag belongs to a tag group '{tgroup}'. This tag means: {description}"""

    _QUERY_WIKI_FEATS_LIST = [
        "lang",
        "tag",
        "key",
        "value",
        "tgroup",
        "redirect_target",
        "description",
    ]
    _QUERY_WIKI_FEATS_STR = ", ".join(_QUERY_WIKI_FEATS_LIST)

    _SQL_TEMPLATE_QUERY_KEY_VALUE = """SELECT {} FROM wikipages where key is ? and value is ? and type is "page";""".format(
        _QUERY_WIKI_FEATS_STR
    )
    _SQL_TEMPLATE_QUERY_KEY = """SELECT {} FROM wikipages where key is ? and value is NULL and type is "page";""".format(
        _QUERY_WIKI_FEATS_STR
    )
    _SQL_TEMPLATE_QUERY_KEY_FUZZY = """SELECT {} FROM wikipages where key like ? and value is NULL and type is "page";""".format(
        _QUERY_WIKI_FEATS_STR
    )

    def __init__(
        self,
        interpret_tasks: list,
        policy: str,
        wiki_db_path: str,
        candidate_elements: gpd.GeoDataFrame = None,
    ) -> None:

        self.interpret_tasks = interpret_tasks
        self.policy = policy
        self.wiki_db_path = wiki_db_path
        self.candidate_elements = candidate_elements
        self.conn_wiki = sqlite3.connect(self.wiki_db_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn_wiki.close()

    def get_working_element_ids(self):

        return self.working_elements["id"].tolist()

    def get_working_elements(self):

        raise NotImplementedError()

    def sort_elements(self):

        raise NotImplementedError()

    def check_working_elements_validity(self) -> bool:

        logging.warning(
            "Checking validity of working elements not implemented yet. Return True by default."
        )
        return True

    def _task_clip(self, num_rows=None, **kwargs):

        if num_rows is None:
            num_rows = len(self.working_elements)

        working_elements = self.working_elements.iloc[:num_rows]

        clipped = []

        for _, element in working_elements.iterrows():
            clip = ""
            if element["clipped"] == True:
                clip = "Some parts of the geometry extended out of this ROI.\r\n"

            clipped.append(clip)

        return clipped

    def interpret_tags(self, num_rows=None):
        """
        This method is used to interpret the tags of the working elements. It will query the wiki database to get the description of the tag.

        Args:
            num_rows (int, optional): the number of rows to interpret. If num_rows is None, all rows will be interpreted. Defaults to None.

        Returns:
            list: a list of interpretations for each element. Each interpretation is a list of prompts for each tag.
        """

        if num_rows is None:
            num_rows = len(self.working_elements)

        working_elements = self.working_elements.iloc[:num_rows]

        interpretations = []
        prompt_template = dict(k=self._TAG_TEMPLATE_K, kv=self._TAG_TEMPLATE_KV)
        c = self.conn_wiki.cursor()
        for _, element in working_elements.iterrows():
            element_prompt = []
            tags = element["tags"]
            for key, value in tags.items():

                cursor = c.execute(self._SQL_TEMPLATE_QUERY_KEY_VALUE, (key, value))
                res = cursor.fetchall()
                no_value = False
                if len(res) < 1:
                    cursor = c.execute(self._SQL_TEMPLATE_QUERY_KEY, (key,))
                    res = cursor.fetchall()
                    no_value = True
                # if len is still smaller than 1, it means there is no match case, try search more starting with the key
                if len(res) < 1:
                    cursor = c.execute(self._SQL_TEMPLATE_QUERY_KEY_FUZZY, (key + "%",))
                    res = cursor.fetchall()
                    no_value = True

                if len(res) < 1:
                    # TODO: solve this anomaly. If there is a key, it has to represent something.
                    continue

                res_df = pd.DataFrame(res, columns=self._QUERY_WIKI_FEATS_LIST)

                # When multi-language is available, English is our first choice.
                res_df["lang_sort"] = res_df["lang"].apply(
                    lambda x: 0 if x == "en" else 1
                )
                res_df.sort_values("lang_sort", ascending=True, inplace=True)
                res_description = res_df[
                    ["tgroup", "redirect_target", "description"]
                ].to_dict("records")[0]

                # generate an individual prompt for a tag's key value pair
                template_ind = "k" if no_value else "kv"
                individual_prompt = prompt_template[template_ind].format(
                    key=key,
                    value=value,
                    tgroup=res_description["tgroup"],
                    description=res_description["description"],
                )
                element_prompt.append(individual_prompt)

            interpretations.append(element_prompt)

        return interpretations

    def interpret_meta(self):

        interpretations = dict()
        for task in self.interpret_tasks:
            _task = task.copy()
            task_name = _task.pop("task")
            interpret_func = getattr(self, "_task_{}".format(task_name))
            interpretations[task_name] = interpret_func(**_task)

        return interpretations

    def plot(self, **kwargs):

        f, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.set_axis_off()
        ax = self.candidate_elements.plot(
            ax=ax, color="white", edgecolor="black", alpha=0.1
        )
        contextily.add_basemap(ax, crs="WGS84")
        return ax

    def plot_on_patch(self, patch_path, color_rgb=None, **kwargs):

        # read the patch image in RGB format for matplotlib
        # cv2.imread returns BGR format, so we need to convert it to RGB
        img = cv2.imread(patch_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_h, im_w, _ = img.shape
        for working_element in self.working_elements.iterrows():
            # extract the geometry of the working element, which is a shapely object
            geom = working_element[1].normed_geometry
            geom_type = geom.geom_type
            # convert the geometry from normalized coordinates to pixel coordinates
            geom = affine_transform(geom, [im_w, 0, 0, -im_h, 0, im_h])

            color = (
                (random() * 255, random() * 255, random() * 255)
                if color_rgb is None
                else color_rgb
            )

            # plot the geometry according to its type
            if geom_type == "Polygon":
                cv2.polylines(
                    img, [np.array(geom.exterior.coords).astype(int)], True, color, 2
                )
            elif geom_type == "LineString":
                cv2.polylines(img, [np.array(geom.coords).astype(int)], False, color, 2)
            elif geom_type == "Point":
                cv2.circle(img, (int(geom.x), int(geom.y)), 5, color, -1)
            elif geom_type == "MultiLineString":
                for line in geom.geoms:
                    cv2.polylines(
                        img, [np.array(line.coords).astype(int)], False, color, 2
                    )
            elif geom_type == "MultiPolygon":
                for poly in geom.geoms:
                    cv2.polylines(
                        img,
                        [np.array(poly.exterior.coords).astype(int)],
                        True,
                        color,
                        2,
                    )
            elif geom_type == "GeometryCollection":
                for geom_i in geom.geoms:
                    if geom_i.geom_type == "Polygon":
                        cv2.polylines(
                            img,
                            [np.array(geom_i.exterior.coords).astype(int)],
                            True,
                            color,
                            2,
                        )
                    elif geom_i.geom_type == "LineString":
                        cv2.polylines(
                            img, [np.array(geom_i.coords).astype(int)], False, color, 2
                        )
                    elif geom_i.geom_type == "Point":
                        cv2.circle(img, (int(geom_i.x), int(geom_i.y)), 5, color, -1)
            else:
                print("Unsupported geometry type: {}".format(geom_type))

        # plot with matplotlib
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_axis_off()
        ax.imshow(img)

        return ax

    def fill_prompt(self, prompt_template: str, **kwargs):
        # fill the prompt template with the given arguments
        # The prompt template should be a string with placeholders in curly braces, 
        # e.g. "The area with the largest size within the ROI is centered at {norm_center}. 
        # Its geometry is {norm_geometry} with a {shape} shape. 
        # Its normalized size is {norm_area} and its real size is {area} square meters."
        return prompt_template.format(**kwargs)


class AreaInterpreter(BaseInterpreter):

    _SHAPE = ["square", "rectangular", "circular", "irregular"]

    def __init__(
        self,
        candidate_areas: gpd.GeoDataFrame,
        wiki_db_path: str,
        interpret_tasks: list = [
            dict(task="center_corse", edge_indicator=False),
            dict(task="shape"),
            dict(task="normed_area", precision=3),
            dict(task="geometry", tolerance=0.6, precision=3),
            dict(task="clip"),
        ],
        policy="largest",
    ) -> None:

        assert policy in ["largest"], "Only largest area is supported for now"

        super().__init__(interpret_tasks, policy, wiki_db_path, candidate_areas)
        self.candidate_areas = candidate_areas
        self.working_elements = self.get_working_elements()
        self.valid_threshold = 0.1

    def sort_elements(self):
        # this method is used to sort the candidate areas based on policy
        sorted_elements = None
        if self.policy == "largest":
            sorted_elements = self.candidate_areas.sort_values(
                "normed_area", ascending=False
            ).reset_index()
        else:
            raise NotImplementedError("Only largest area is supported for now")

        return sorted_elements

    def get_working_elements(self):

        working_elements = self.sort_elements().iloc[[0]]

        return working_elements

    def _task_center_corse(self, edge_indicator=False, **kwargs):

        coord_mapper = Coord2NineGridMapper(edge_indicator)
        center_pt = self.working_elements["normed_center"][0]
        center_pt = (center_pt.x, center_pt.y)
        corse_pos = coord_mapper.convert(center_pt)

        return corse_pos

    def _task_shape(self, **kwargs):

        geom = self.working_elements["normed_geometry"][0]

        if geom.geom_type == "MultiPolygon":
            return "Multi-Polygonal"
        elif geom.geom_type == "GeometryCollection":
            return "Multi-Geometric"

        circle_area = geom.length**2 / (4 * np.pi)
        area_perim_ratio = geom.area / geom.length
        if (
            area_perim_ratio > 0.9 * geom.length
            and area_perim_ratio < 1.1 * geom.length
        ):
            s = 0
        elif len(geom.exterior.coords) < 8:
            s = 1
        elif circle_area > 0.9 * geom.area and circle_area < 1.1 * geom.area:
            s = 2
        else:
            s = -1

        return self._SHAPE[s]

    def _task_normed_area(self, precision=3, **kwargs):

        area = self.working_elements["normed_area"][0]
        return "{:.{}f}".format(area, precision)

    def _task_geometry(self, tolerance=0.5, precision=3, **kwargs):

        geom = self.working_elements["normed_geometry"][0]

        geom = geom.simplify(tolerance, preserve_topology=True)

        geom_type = geom.geom_type

        if geom_type == "Polygon":
            geom = [geom]
        else:
            geom = list(geom.geoms)

        geom_str = []
        for g in geom:
            coords = list(g.exterior.coords)
            geom_single = ""
            vertex = []
            for coord in coords:
                x_pos, y_pos = coord
                vertex.append(
                    "({x_pos:.{prec}f}, {y_pos:.{prec}f})".format(
                        x_pos=x_pos, y_pos=y_pos, prec=precision
                    )
                )
            geom_single = ", ".join(vertex)
            geom_single = "[" + geom_single + "]"
            geom_str.append(geom_single)

        out_str = ", ".join(geom_str)

        out_str = "{" + out_str + "}"

        return out_str

    def check_working_elements_validity(self) -> bool:
        """
        This method is used to check the validity of the working elements.
        If the working elements is smaller than a certain threshold,
        it is considered invalid.

        Returns:
            bool: True if the working elements is valid, False otherwise.
        """

        validity = True

        if self.working_elements["normed_area"][0] < self.valid_threshold:
            validity = False

        return validity

    def _task_clip(self, **kwargs):
        return super()._task_clip(1, **kwargs)[0]

    def interpret_tags(self):

        tags = super().interpret_tags(1)
        return tags[0]

    def plot(self, **kwargs):

        ax = super().plot(**kwargs)
        self.working_elements.plot(ax=ax, color="red", alpha=0.5)
        return ax


# This class is used for interpret the raw geometry of the nonearea into natural language
class NoneAreaInterpreter(BaseInterpreter):

    _SINUOSITY = dict(
        straight=(1, 1.0005),
        curved=(1.0005, 1.5),
        twisted=(1.5, np.inf),
        closed=(np.inf,),
        broken=(0, 1),
    )

    _ORIENTATION = dict(
        W_E=((-np.pi / 6, np.pi / 6),),
        S_N=((-np.pi / 2, -np.pi / 3), (np.pi / 3, np.pi / 2)),
        SW_NE=((np.pi / 6, np.pi / 3),),
        NW_SE=((-np.pi / 3, -np.pi / 6),),
    )

    def __init__(
        self,
        candidate_noneareas: gpd.GeoDataFrame,
        wiki_db_path: str,
        interpret_tasks: list = [
            dict(task="endpoints_corse", edge_indicator=True),
            dict(task="sinuosity"),
            dict(task="normed_length", precision=3),
            dict(task="length", precision=0),
            dict(task="orientation"),
            dict(task="geometry", tolerance=0.6, precision=3),
            dict(task="clip"),
        ],
        policy="longgest_w_most_tags",
    ) -> None:

        assert policy in [
            "longgest_w_most_tags"
        ], "Only longgest as well as the most tags nonearea is supported for now"

        super().__init__(interpret_tasks, policy, wiki_db_path, candidate_noneareas)
        self.candidate_noneareas = candidate_noneareas
        self.working_elements = self.get_working_elements()

    def sort_elements(self):
        # this method is used to sort the candidate areas based on policy
        sorted_elements = None
        if self.policy == "longgest_w_most_tags":
            sorted_elements = self.candidate_noneareas.sort_values(
                "normed_length", ascending=False
            ).reset_index()
            # get the top 3 noneareas with the most tags, note that there may be fewer than 3 noneareas available
            sorted_elements = sorted_elements.iloc[:3]
            sorted_elements["num_tags"] = sorted_elements["tags"].apply(
                lambda x: len(dict(x))
            )
            sorted_elements = sorted_elements.sort_values(
                "num_tags", ascending=False
            ).reset_index()
        else:
            raise NotImplementedError("Only longgest nonearea is supported for now")

        return sorted_elements

    def get_working_elements(self):

        working_elements = self.sort_elements().iloc[[0]]

        return working_elements

    def _task_endpoints_corse(self, edge_indicator=False, **kwargs):

        geom = self.working_elements["normed_geometry"][0]
        start_pt = geom.interpolate(0)
        end_pt = geom.interpolate(1, normalized=True)

        if geom.geom_type == "LineString":
            start_pt = geom.interpolate(0)
            end_pt = geom.interpolate(1, normalized=True)
        elif geom.geom_type == "MultiLineString":
            # find the longest line segment to determine the start and end points
            max_length = 0
            for line in geom.geoms:
                if line.length > max_length:
                    start_pt = line.interpolate(0)
                    end_pt = line.interpolate(1, normalized=True)
                    max_length = line.length
        else:
            return "The geometry is not a line or a multi-line"

        interpretation = []
        coord_mapper = Coord2NineGridMapper(edge_indicator)
        pts = tuple(start_pt.coords)[0], tuple(end_pt.coords)[0]

        for pt in pts:
            corse_pos = coord_mapper.convert(pt)
            interpretation.append(corse_pos)

        return f"One endpoint locates at {interpretation[0]}, the other at {interpretation[1]}"

    def _task_sinuosity(self, **kwargs):

        def get_sinuosity_single(line):
            start_pt = line.interpolate(0)
            end_pt = line.interpolate(1, normalized=True)
            straight_dist = start_pt.distance(end_pt)
            sinuosity = np.divide(line.length, straight_dist)
            return sinuosity

        geom = self.working_elements["normed_geometry"][0]

        if geom.geom_type == "LineString":
            sinuosity = get_sinuosity_single(geom)
        elif geom.geom_type == "MultiLineString":
            # if there are multiple line segments, calculate the sinuosity of each segment and take the maximum
            sinuosity = []
            for line in geom.geoms:
                sinuosity.append(get_sinuosity_single(line))
            sinuosity = max(sinuosity)
        else:
            return "The geometry is not a line or a multi-line"

        for k, v in self._SINUOSITY.items():
            if len(v) == 1:
                if sinuosity == v:
                    return k
            else:
                if sinuosity >= v[0] and sinuosity < v[1]:
                    return k

    def _task_normed_length(self, precision=3, **kwargs):

        length = self.working_elements["normed_length"][0]
        return "{:.{}f}".format(length, precision)

    def _task_length(self, precision=3, **kwargs):

        length = self.working_elements["length"][0]
        return "{:.{}f}".format(length, precision)

    # example of how to use the orientation function
    # def interpret_orientation(self):

    #     orientation = "Cannot be determined"

    #     if not self.geom_type == "LineString":
    #         return orientation

    #     pt1, pt2 = tuple(self.start_pt.coords)[0], tuple(self.end_pt.coords)[0]
    #     angle = np.arctan(np.divide((pt2[1] - pt1[1]), (pt2[0] - pt1[0])))
    #     if angle == np.pi/2:
    #         angle = -np.pi/2

    #     for k, ranges in self._ORIENTATION.items():
    #         for r in ranges:
    #             if angle >= r[0] and angle < r[1]:
    #                 orientation=k
    #                 break

    #     return orientation

    def _task_orientation(self, **kwargs):

        orientation = "Cannot be determined"

        geom = self.working_elements["normed_geometry"][0]

        if geom.geom_type == "LineString":
            start_pt = geom.interpolate(0)
            end_pt = geom.interpolate(1, normalized=True)

            angle = np.arctan(
                np.divide((end_pt.y - start_pt.y), (end_pt.x - start_pt.x))
            )
            if angle == np.pi / 2:
                angle = -np.pi / 2
        elif geom.geom_type == "MultiLineString":
            # if there are multiple line segments, determine the orientation with the longest segment
            max_length = 0
            for line in geom.geoms:
                if line.length > max_length:
                    start_pt = line.interpolate(0)
                    end_pt = line.interpolate(1, normalized=True)
                    max_length = line.length
                    angle = np.arctan(
                        np.divide((end_pt.y - start_pt.y), (end_pt.x - start_pt.x))
                    )
                    if angle == np.pi / 2:
                        angle = -np.pi / 2
        else:
            return "The geometry is not a line or a multi-line"

        for k, ranges in self._ORIENTATION.items():
            for r in ranges:
                if angle >= r[0] and angle < r[1]:
                    orientation = k
                    break

        return orientation

    def _task_clip(self, **kwargs):
        return super()._task_clip(1, **kwargs)[0]

    def _task_geometry(self, tolerance=0.5, precision=3, **kwargs):

        geom = self.working_elements["normed_geometry"][0]

        geom = geom.simplify(tolerance, preserve_topology=True)

        geom_type = geom.geom_type

        if geom_type == "LineString":
            geom = [geom]
        else:
            geom = list(geom.geoms)

        geom_str = []
        for g in geom:
            coords = list(g.coords)
            geom_single = ""
            vertex = []
            for coord in coords:
                x_pos, y_pos = coord
                vertex.append(
                    "({x_pos:.{prec}f}, {y_pos:.{prec}f})".format(
                        x_pos=x_pos, y_pos=y_pos, prec=precision
                    )
                )
            geom_single = ", ".join(vertex)
            geom_single = "[" + geom_single + "]"
            geom_str.append(geom_single)

        out_str = ", ".join(geom_str)

        out_str = "{" + out_str + "}"

        return out_str

    def interpret_tags(self):

        tags = super().interpret_tags(1)
        return tags[0]

    def plot(self, **kwargs):

        ax = super().plot(**kwargs)
        self.working_elements.plot(ax=ax, color="red", alpha=0.5)
        return ax

    def check_working_elements_validity(self) -> bool:

        return True


class AreaRelationInterpreter(object):

    _ORIENTATION = dict(
        left=((157.5, 180), (-180, -157.5)),
        upper_left=((112.5, 157.5),),
        upper=((67.5, 112.5),),
        upper_right=((22.5, 67.5),),
        right=((-22.5, 22.5),),
        lower_right=((-67.5, -22.5),),
        lower=((-112.5, -67.5),),
        lower_left=((-157.5, -112.5),),
    )

    """
    distance specifics:
    Adjacent (0, 0.01): This range suggests that the elements are extremely near to each other, almost touching or immediately next to each other.
    Nearby (0.01, 0.05): Elements within this range are in close proximity but not immediately adjacent. They are easily reachable from one another.
    Close (0.05, 0.2): This indicates a moderate distance between elements. They are not far from each other, but there's a noticeable gap.
    Intermediate (0.2, 0.4): This range suggests an average distance, neither too close nor too far. It's a middle ground.
    Distant (0.4, 0.6): Elements are quite a distance apart, requiring more effort or time to traverse between them.
    Remote (0.6, 2): This indicates a significant distance between the elements, suggesting they are at opposite ends of the region of interest or very widely spaced.
    """
    _DISTANCE = dict(
        adjacent=(0, 0.01),
        nearby=(0.01, 0.05),
        close=(0.05, 0.2),
        intermediate=(0.2, 0.4),
        distant=(0.4, 0.6),
        remote=(0.6, 2),
    )

    def __init__(self, candidate_areas, n_neighbors=3):
        self.candidate_areas = candidate_areas
        self.n_neighbors = n_neighbors
        self.coord_mapper = Coord2NineGridMapper(edge_indicator=False)
        self.w_knn = KNN.from_dataframe(self.candidate_areas, k=n_neighbors)
        self.task_areas = self._get_task_areas()
        self.focal_area = self._get_focal_area()
        self.nearby_areas = self._get_nearby_areas()
        self._get_positional_attributes()

    def _get_task_areas(self):
        # TODO: Make sure there are at least three areas
        rand_inds = np.random.permutation(len(self.candidate_areas))[
            : np.random.randint(2, 4)
        ]
        selected_areas = self.candidate_areas.iloc[rand_inds]
        focus_area = selected_areas.sort_values("normed_area", ascending=False).iloc[
            [0]
        ]
        nearby_indx = self.w_knn.neighbors[focus_area.index[0]]
        task_area_indx = nearby_indx + [focus_area.index[0]]
        task_areas = self.candidate_areas.loc[task_area_indx]

        return task_areas

    def _get_focal_area(self):

        return self.task_areas.sort_values("normed_area", ascending=False).iloc[[0]]

    def _get_nearby_areas(self):

        return self.task_areas.sort_values("normed_area", ascending=False).iloc[1:]

    def _get_positional_attributes(self):

        self.nearby_areas["within"] = self.nearby_areas["center"].within(
            self.focal_area.iloc[0].geometry
        )
        self.nearby_areas["distance2focal"] = self.focal_area.iloc[
            0
        ].normed_center.distance(self.nearby_areas["normed_center"])
        self.nearby_areas[self.nearby_areas["distance2focal"] == 180] = -180

        def get_relative_position(center1, center2):
            return (
                np.arctan2(center2[1] - center1[1], center2[0] - center1[0])
                / np.pi
                * 180
            )

        self.nearby_areas["postion2focal"] = self.nearby_areas["normed_center"].apply(
            lambda x: get_relative_position(
                np.array(self.focal_area.iloc[0].normed_center.coords[0]),
                np.array(x.coords[0]),
            )
        )

    def interpret_focal_area_meta(self):
        # prepare the information needed in a prompt template and save them into a dict
        area_info = dict()
        center_pos = tuple(self.focal_area.iloc[0].normed_center.coords[0])
        # drop the last comma
        corse_pos = self.coord_mapper.convert(center_pos)[:-1]
        corse_pos = "-".join(corse_pos.split(","))
        area_info.update(corse_pos=corse_pos)
        area_info.update(center_x=center_pos[0], center_y=center_pos[1])
        area_info.update(area=self.focal_area.iloc[0].normed_area)

        return area_info

    def interpret_nearby_areas_meta(self):
        # prepare the information needed in a prompt template and save them into dicts
        area_infos = []
        for i, row in self.nearby_areas.iterrows():
            inclusion = "within" if row["within"] else "outside"
            for k, range in self._DISTANCE.items():
                if (
                    row["distance2focal"] < range[1]
                    and row["distance2focal"] >= range[0]
                ):
                    distance = k
                    break
            for k, ranges in self._ORIENTATION.items():
                for r in ranges:
                    if row["postion2focal"] < r[1] and row["postion2focal"] >= r[0]:
                        orientation = "-".join(k.split("_"))
                        break
            area = row["normed_area"]
            area_info = dict(
                inclusion=inclusion,
                distance=distance,
                orientation=orientation,
                area=area,
            )
            area_infos.append(area_info)

        return area_infos

    def get_focal_area_tags(self):

        return self.focal_area.iloc[0].tags

    def get_nearby_areas_tags(self):

        return self.nearby_areas.tags.tolist()

    def plot_task_areas(self, focal_color="red", nearby_color="blue"):

        f, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.set_axis_off()
        ax = self.candidate_areas.plot(
            ax=ax, color="white", edgecolor="black", alpha=0.1
        )
        ax = self.focal_area.plot(
            ax=ax, color=focal_color, edgecolor="black", alpha=0.5
        )
        ax = self.nearby_areas.plot(
            ax=ax, color=nearby_color, edgecolor="black", alpha=0.5
        )
        contextily.add_basemap(ax, crs="WGS84")

        return f, ax


def task2_interpretor(
    meta, osm_elements, wiki_db_path, prompt_template, interpret_configs: dict
):

    center_x, center_y, dimension, download_time, patch_crs, scale = meta[2:]

    osm_working_area = get_osm_working_boundary(
        (center_x, center_y), dimension, download_time, patch_crs, scale
    )
    working_area = [
        osm_working_area[1],
        osm_working_area[0],
        osm_working_area[3],
        osm_working_area[2],
    ]

    osm_area_gdf = gpd.GeoDataFrame(osm_elements, columns=["id", "tags", "geometry"])
    osm_area_gdf["tags"] = osm_area_gdf["tags"].apply(lambda x: json.loads(x))
    osm_area_gdf["geometry"] = osm_area_gdf["geometry"].apply(lambda x: from_wkt(x))
    osm_area_gdf.set_geometry("geometry", inplace=True)
    osm_area_gdf.set_crs(epsg=4326, inplace=True)

    masked_osm_area_gdf = osm_area_gdf.clip(box(*working_area))
    masked_osm_area_gdf["clipped"] = (
        osm_area_gdf["geometry"].within(box(*working_area)).apply(lambda x: not x)
    )

    # TODO: use dynamic config
    # normalizer_config = interpret_configs['normalizer']
    normalizer_config = dict(proj_epsg=2062)
    normalizer = GeometryNormalizer(working_area, **normalizer_config)
    normed_area_gdf = normalizer.area_normalization(masked_osm_area_gdf)

    # TODO: use dynamic config
    interpreter_config = interpret_configs.get("interpreter", dict())
    with AreaInterpreter(
        normed_area_gdf, wiki_db_path, **interpreter_config
    ) as interpreter:

        # check validity of the working elements
        if not interpreter.check_working_elements_validity():
            return None

        prompt_vars = dict()

        meta_interpretation = interpreter.interpret_meta()
        tag_interpretation = interpreter.interpret_tags()

        tags = """"""
        for i, tag in enumerate(tag_interpretation):
            tags += f"Tag {i+1}. {tag}\n"

        prompt_vars.update(dict(tags=tags))

        prompt_vars.update(meta_interpretation)
        prompt_vars.update(dict(tag_interpretation=tag_interpretation))
        prompt = interpreter.fill_prompt(prompt_template, **prompt_vars)
        osm_ids = interpreter.get_working_element_ids()

    # remove duplicate '.'
    prompt = ".".join(prompt.split(".."))

    return dict(prompt=prompt, osm_ids=osm_ids)


def task3_interpretor(
    meta, osm_elements, wiki_db_path, prompt_template, interpret_configs
):

    center_x, center_y, dimension, download_time, patch_crs, scale = meta[2:]

    osm_working_area = get_osm_working_boundary(
        (center_x, center_y), dimension, download_time, patch_crs, scale
    )
    working_area = [
        osm_working_area[1],
        osm_working_area[0],
        osm_working_area[3],
        osm_working_area[2],
    ]

    osm_nonearea_gdf = gpd.GeoDataFrame(
        osm_elements, columns=["id", "tags", "geometry"]
    )
    osm_nonearea_gdf["tags"] = osm_nonearea_gdf["tags"].apply(lambda x: json.loads(x))
    osm_nonearea_gdf["geometry"] = osm_nonearea_gdf["geometry"].apply(
        lambda x: from_wkt(x)
    )
    osm_nonearea_gdf.set_geometry("geometry", inplace=True)
    osm_nonearea_gdf.set_crs(epsg=4326, inplace=True)

    masked_osm_nonearea_gdf = osm_nonearea_gdf.clip(box(*working_area))
    masked_osm_nonearea_gdf["clipped"] = (
        osm_nonearea_gdf["geometry"].within(box(*working_area)).apply(lambda x: not x)
    )

    # TODO: use dynamic config
    # normalizer_config = interpret_configs['normalizer']
    normalizer_config = dict(proj_epsg=2062)
    normalizer = GeometryNormalizer(working_area, **normalizer_config)
    normed_area_gdf = normalizer.nonearea_normalization(masked_osm_nonearea_gdf)

    # TODO: use dynamic config
    # interpreter_config = interpret_configs['interpreter']
    interpreter_config = dict()
    with NoneAreaInterpreter(
        normed_area_gdf, wiki_db_path, **interpreter_config
    ) as interpreter:

        # check validity of the working elements
        if not interpreter.check_working_elements_validity():
            return None

        prompt_vars = dict()

        meta_interpretation = interpreter.interpret_meta()
        tag_interpretation = interpreter.interpret_tags()

        tags = """"""
        for i, tag in enumerate(tag_interpretation):
            tags += f"Tag {i+1}. {tag}\n"

        prompt_vars.update(dict(tags=tags))

        prompt_vars.update(meta_interpretation)
        prompt_vars.update(dict(tag_interpretation=tag_interpretation))
        prompt = interpreter.fill_prompt(prompt_template, **prompt_vars)
        osm_ids = interpreter.get_working_element_ids()

    # remove duplicate '.'
    prompt = ".".join(prompt.split(".."))

    return dict(prompt=prompt, osm_ids=osm_ids)


def annotation_parser_prompt1_annotator1(annotation):

    task_split = annotation.split("###Task###")
    pure_text = task_split[1].split("Caption:\n")[1].strip()

    return pure_text


def annotation_parser_prompt4_annotator(annotation):
    task_split = annotation.split("###TASK###")
    pure_text = task_split[1].split("Revised:\n")[1].strip()

    return pure_text
