"""
core/boundary.py
================
Site boundary loaded from a GeoJSON file (WGS84 lon/lat coordinates).

Typical workflow for the user:
  1. Draw a polygon on geojson.io (including forbidden zones as holes).
  2. Export as GeoJSON → place in the config/ folder.
  3. Point boundary_geojson in the case YAML to that file.

Pipeline inside this module:
  1. Read the GeoJSON polygon (lon/lat in degrees).
  2. Project to local metres using an Azimuthal Equidistant projection
     centred at the polygon centroid  →  centroid becomes (0, 0).
  3. Expose a simple interface used by the optimisers and the plotter.

Accepted GeoJSON structures (all exported naturally by geojson.io):
  - { "type": "Feature", "geometry": { "type": "Polygon", ... } }
  - { "type": "FeatureCollection", "features": [<single feature>] }
  - { "type": "Polygon", ... }   (raw geometry)

Polygons with holes (forbidden zones drawn as inner rings) are fully
supported in all operations.
"""

import json
import random

from shapely.geometry import Point, shape
from shapely.ops import transform, nearest_points, unary_union
import pyproj


class SiteBoundary:
    """
    Site boundary in local Cartesian metres, centred at (0, 0).

    Attributes
    ----------
    bbox : tuple
        (xmin, ymin, xmax, ymax) in metres.
    area_km2 : float
        Site area (holes excluded) in km².
    """

    def __init__(self, polygon_m, lon0=0.0, lat0=0.0):
        self._poly = polygon_m
        self.bbox = polygon_m.bounds          # (xmin, ymin, xmax, ymax)
        self.area_km2 = polygon_m.area / 1e6
        self.substation_pos = None            # Populated if a Point is found in GeoJSON
        self.lon0 = lon0
        self.lat0 = lat0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_geojson(cls, geojson_path):
        """
        Load a GeoJSON file and return a SiteBoundary in local metres.

        Parameters
        ----------
        geojson_path : str
            Path to the .geojson file (WGS84 coordinates, as exported
            by geojson.io, QGIS, Felt, etc.)
        """
        with open(geojson_path, "r") as f:
            data = json.load(f)

        geom_dicts = _extract_geometries(data)
        shapes = [shape(g) for g in geom_dicts if g is not None]
        
        polygons = []
        points = []
        for s in shapes:
            if s.geom_type == "Polygon":
                polygons.append(s)
            elif s.geom_type == "MultiPolygon":
                polygons.extend(list(s.geoms))
            elif s.geom_type == "Point":
                points.append(s)
                
        if not polygons:
            raise ValueError(
                "No Polygon geometries found in the GeoJSON. "
                "Draw at least one polygon in geojson.io."
            )

        polygons.sort(key=lambda p: p.area, reverse=True)
        
        base_regions = [polygons[0]]
        holes = []
        
        for p in polygons[1:]:
            # If p is mostly contained within any base region, it's a hole
            is_hole = False
            for base in base_regions:
                if base.contains(p) or base.intersection(p).area > 0.9 * p.area:
                    is_hole = True
                    break
            
            if is_hole:
                holes.append(p)
            else:
                base_regions.append(p)

        polygon_lonlat = unary_union(base_regions)
        for hole in holes:
            polygon_lonlat = polygon_lonlat.difference(hole)

        if polygon_lonlat.geom_type not in ["Polygon", "MultiPolygon"]:
            raise ValueError(
                f"Resulting geometry must be Polygon or MultiPolygon, got "
                f"'{polygon_lonlat.geom_type}'."
            )

        # Azimuthal Equidistant projection centred at the polygon centroid.
        # Distances are accurate to within ~0.1 % for sites up to ~500 km.
        lon0, lat0 = polygon_lonlat.centroid.x, polygon_lonlat.centroid.y
        proj_str = (
            f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} "
            "+datum=WGS84 +units=m +no_defs"
        )
        transformer = pyproj.Transformer.from_crs(
            "EPSG:4326", proj_str, always_xy=True
        )
        polygon_m = transform(transformer.transform, polygon_lonlat)
        
        boundary = cls(polygon_m, lon0, lat0)
        
        if points:
            if len(points) > 1:
                raise ValueError(
                    "Multiple Point geometries found in the GeoJSON. "
                    "The simulator currently supports only one substation."
                )
            # Transform the first point found in the GeoJSON
            pt_lon, pt_lat = points[0].x, points[0].y
            pt_x, pt_y = transformer.transform(pt_lon, pt_lat)
            boundary.substation_pos = [float(pt_x), float(pt_y)]
        
        return boundary

    # ------------------------------------------------------------------
    # Geometric operations (used by the optimisers)
    # ------------------------------------------------------------------

    def contains(self, x, y):
        """
        Return True if (x, y) is inside the valid site area.
        Points inside holes (forbidden zones) return False.
        """
        return self._poly.contains(Point(x, y))

    def enforce(self, x, y):
        """
        If (x, y) is outside the valid area, return the nearest point on
        the polygon boundary.
        """
        pt = Point(x, y)
        if self._poly.contains(pt):
            return x, y
        _, nearest = nearest_points(pt, self._poly.boundary)
        return nearest.x, nearest.y

    def random_point(self):
        """
        Return a uniformly random point inside the valid area.
        Uses rejection sampling — robust for any polygon shape or holes.
        Raises RuntimeError if the polygon is degenerate.
        """
        xmin, ymin, xmax, ymax = self.bbox
        for _ in range(100_000):
            x = random.uniform(xmin, xmax)
            y = random.uniform(ymin, ymax)
            if self._poly.contains(Point(x, y)):
                return x, y
        raise RuntimeError(
            "Could not sample a valid point inside the boundary after many "
            "attempts. Check that your GeoJSON polygon is not degenerate."
        )

    def crosses_hole(self, line):
        """
        Return True if the LineString crosses any hole (forbidden zone).
        It is allowed to go outside the exterior boundary, but not through holes.
        """
        from shapely.geometry import Polygon
        if self._poly.geom_type == "Polygon":
            polys = [self._poly]
        elif self._poly.geom_type == "MultiPolygon":
            polys = list(self._poly.geoms)
        else:
            polys = []
            
        for p in polys:
            for interior in p.interiors:
                hole_poly = Polygon(interior)
                intersection = hole_poly.intersection(line)
                if not intersection.is_empty and intersection.length > 1e-3:
                    return True
        return False

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def to_patch(self, **kwargs):
        """
        Return a matplotlib PathPatch for the boundary.
        Holes are rendered correctly as transparent cut-outs.

        Pass any matplotlib Patch keyword arguments to override defaults.
        """
        from matplotlib.patches import PathPatch
        from matplotlib.path import Path

        verts, codes = [], []

        def _ring(ring):
            coords = list(ring.coords)
            verts.extend(coords)
            codes.append(Path.MOVETO)
            codes.extend([Path.LINETO] * (len(coords) - 2))
            codes.append(Path.CLOSEPOLY)

        polys = self._poly.geoms if self._poly.geom_type == "MultiPolygon" else [self._poly]
        for p in polys:
            _ring(p.exterior)
            for interior in p.interiors:
                _ring(interior)

        defaults = dict(
            fill=False, linestyle="--", edgecolor="white", alpha=0.4, lw=1.5
        )
        defaults.update(kwargs)
        return PathPatch(Path(verts, codes), **defaults)


# ----------------------------------------------------------------------
# Internal helper
# ----------------------------------------------------------------------

def _extract_geometries(data):
    """
    Extract a list of geometry dicts from any valid GeoJSON top-level structure.
    """
    t = data.get("type", "")
    if t == "Feature":
        return [data.get("geometry")]
    elif t == "FeatureCollection":
        return [f.get("geometry") for f in data.get("features", [])]
    else:
        # Raw geometry object
        if t == "GeometryCollection":
            return data.get("geometries", [])
        return [data]
