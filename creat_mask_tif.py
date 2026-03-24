#!/usr/bin/env python3
"""
split_shp_to_village_tifs.py

Reads a village boundary shapefile (one vector containing many village polygons) and
writes one GeoTIFF per village where pixels inside the polygon are burned with a
value (default 1) and outside are 0. Useful to generate per-village masks.

Usage examples:
  # shapefile path is a directory containing the .shp and supporting files
  python3 split_shp_to_village_tifs.py \
    --shp_dir "/home/asta/poppy/Village Boundary Database_SHAPEFILE__MADHYA PRADESH_MANDSAUR" \
    --out_dir /home/asta/poppy/village_masks --resolution 30

  # or use a reference raster so all outputs align with it
  python3 split_shp_to_village_tifs.py \
    --shp /path/to/villages.shp --ref_raster /path/to/reference.tif --out_dir ./village_masks

Options:
  --shp or --shp_dir : path to .shp file or directory containing shapefile
  --out_dir          : where to save per-village tifs (default: ./village_tifs)
  --ref_raster       : optional reference raster to align outputs (transform, crs, shape)
  --resolution       : pixel size in units of shapefile CRS (used if no ref_raster). If shapefile CRS is geographic, pass degrees.
  --id_field         : attribute to use for id/name. Default tries common fields: ['VILLAGE','NAME','NAME_1','village','id','gid']
  --burn_value       : value to write inside polygons (default 1)
  --dtype            : output dtype (uint8 or uint16)
  --overwrite        : overwrite existing outputs

Notes:
- If neither --ref_raster nor --resolution are provided the script will attempt
  to use a sensible default resolution of 30 (units of vector CRS).
- Output tif CRS will match the reference raster if given, otherwise the vector CRS.

"""

from pathlib import Path
import argparse
import os
import sys
import math
import logging
import re

import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import mapping


LOG = logging.getLogger("split_shp")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def _find_shapefile_in_dir(d: Path) -> Path:
    # return first .shp found
    for ext in (".shp",):
        for p in sorted(d.glob(f"*{ext}")):
            return p
    return None


def _safe_name(s: str, max_len: int = 80) -> str:
    if s is None:
        return "no_name"
    s = str(s)
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-]", "", s)
    return s[:max_len]


def write_mask_from_geometry(geom, out_tif: Path, transform, width, height, crs, burn_value=1, dtype="uint8", overwrite=False):
    if out_tif.exists() and not overwrite:
        LOG.info(f"Skipping existing {out_tif}")
        return
    shapes = [(mapping(geom), burn_value)]
    out_arr = rasterize(shapes, out_shape=(height, width), transform=transform, fill=0, dtype=dtype)
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 1,
        'dtype': dtype,
        'crs': crs,
        'transform': transform,
        'compress': 'deflate'
    }
    with rasterio.open(out_tif, 'w', **meta) as dst:
        dst.write(out_arr, 1)
    LOG.info(f"Wrote {out_tif} (shape={width}x{height})")


def process_shapefile(shp_path: Path, out_dir: Path, ref_raster: Path = None, resolution: float = None, id_field: str = None, burn_value: int = 1, dtype: str = 'uint8', overwrite: bool = False):
    # locate shapefile if dir supplied
    if shp_path.is_dir():
        shp_file = _find_shapefile_in_dir(shp_path)
        if shp_file is None:
            raise FileNotFoundError(f"No .shp file found in {shp_path}")
    else:
        shp_file = shp_path
    LOG.info(f"Reading shapefile {shp_file}")
    gdf = gpd.read_file(shp_file)
    if gdf.empty:
        LOG.error("Shapefile contains no features")
        return

    # decide id/name field
    if id_field is None:
        candidates = ['VILLAGE', 'VILLAGE_NAME', 'NAME', 'NAME_1', 'village', 'name', 'id', 'gid']
        for c in candidates:
            if c in gdf.columns:
                id_field = c
                break
    LOG.info(f"Using id_field={id_field}")

    # Reference raster settings
    ref = None
    if ref_raster is not None:
        ref = rasterio.open(ref_raster)
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_width = ref.width
        ref_height = ref.height
        LOG.info(f"Using reference raster {ref_raster} (crs={ref_crs})")
    else:
        ref_crs = gdf.crs
        ref_transform = None
        ref_width = None
        ref_height = None
        if resolution is None:
            resolution = 30.0
            LOG.info(f"No resolution provided; defaulting to {resolution}")

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(gdf)
    LOG.info(f"Found {total} features; writing masks to {out_dir}")

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            LOG.warning(f"Skipping empty geometry at index {idx}")
            continue
        # choose name
        if id_field and id_field in row and row[id_field] is not None:
            name = _safe_name(row[id_field])
        else:
            name = f"village_{idx}"
        out_tif = out_dir / f"{name}.tif"

        # if using reference raster, reproject geometry to raster CRS and rasterize to ref grid
        if ref is not None:
            try:
                geom_proj = geom.to_crs(ref.crs) if hasattr(geom, 'to_crs') else gpd.GeoSeries([geom], crs=gdf.crs).to_crs(ref.crs).iloc[0]
            except Exception:
                # use geopandas to reproject
                geom_proj = gpd.GeoSeries([geom], crs=gdf.crs).to_crs(ref.crs).iloc[0]
            write_mask_from_geometry(geom_proj, out_tif, ref_transform, ref_width, ref_height, ref_crs, burn_value=burn_value, dtype=dtype, overwrite=overwrite)
        else:
            # compute bounds for this geometry and create a transform at requested resolution
            minx, miny, maxx, maxy = geom.bounds
            # add tiny pad to ensure polygon fits within raster
            pad_x = (maxx - minx) * 0.02 if (maxx - minx) > 0 else 1.0
            pad_y = (maxy - miny) * 0.02 if (maxy - miny) > 0 else 1.0
            minx -= pad_x
            maxx += pad_x
            miny -= pad_y
            maxy += pad_y
            width = int(math.ceil((maxx - minx) / resolution))
            height = int(math.ceil((maxy - miny) / resolution))
            if width <= 0 or height <= 0:
                LOG.warning(f"Invalid raster size for {name}, skipping")
                continue
            transform = from_bounds(minx, miny, maxx, maxy, width, height)
            write_mask_from_geometry(geom, out_tif, transform, width, height, gdf.crs, burn_value=burn_value, dtype=dtype, overwrite=overwrite)

    if ref is not None:
        ref.close()
    LOG.info("Done")


def parse_args():
    p = argparse.ArgumentParser(description="Split village shapefile into per-village GeoTIFF masks")
    p.add_argument('--shp', help='Path to a .shp file (or directory containing the shapefile)')
    p.add_argument('--shp_dir', help='Directory containing shapefile (alternative to --shp)')
    p.add_argument('--out_dir', help='Output directory', default='./village_tifs')
    p.add_argument('--ref_raster', help='Reference raster to align outputs (optional)')
    p.add_argument('--resolution', type=float, help='Pixel size (units of shapefile CRS). Used when no ref_raster provided')
    p.add_argument('--id_field', help='Attribute name to use for output filenames')
    p.add_argument('--burn_value', type=int, default=1, help='Value to burn inside village polygon')
    p.add_argument('--dtype', choices=['uint8','uint16'], default='uint8')
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    shp_input = args.shp if args.shp else args.shp_dir
    if not shp_input:
        LOG.error("Please provide --shp or --shp_dir pointing to the village shapefile or folder")
        sys.exit(1)
    process_shapefile(Path(shp_input), Path(args.out_dir), ref_raster=Path(args.ref_raster) if args.ref_raster else None, resolution=args.resolution, id_field=args.id_field, burn_value=args.burn_value, dtype=args.dtype, overwrite=args.overwrite)
