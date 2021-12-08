#!/usr/bin/env python3

import os
import argparse
import isce3
from nisar.products.readers import SLC
from osgeo import gdal, osr
threshold = 1.0e-7
numiter = 25
extraiter = 10
lines_per_block = 1000
threshold_geo2rdr = 1e-8
iteration_geo2rdr = 25
dem_block_margin = 0.1


EXAMPLE = '''

  Example command:

  geocode_shadow_layover.py -r data/SanAnd_05518_12018_000_120419_L090_CX_143_03.h5 -d data/dem.tif -g gslc_ref/gslc.h5 -f A -p HH -o shadow_layover.geo



'''
def create_parser():
    parser = argparse.ArgumentParser(description='Create shadow layover mask over radar grid coordinates and geocode to a geododed grid',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=EXAMPLE)
    parser.add_argument('-r', '--rslc', type = str, default = None, dest = 'rslc', help = 'The NISAR HDF5 RSLC product.')
    parser.add_argument('-f', '--frequency', type = str, default = None, dest = 'frequency', help = 'The frequency to use A or B')
    parser.add_argument('-p', '--polarization', type = str, default = None, dest = 'polarization', help = 'The polarization to use')
    parser.add_argument('-d', '--dem', type = str, default = None, dest = 'dem', help = 'A Digital Elevation Model (DEM) that covers the region of interest.')
    parser.add_argument('-g', '--gslc', type = str, default = None, dest = 'gslc', help = 'geocoded SLC')
    parser.add_argument('-o', '--output', type = str, default = None, dest = 'output', help = 'Output Geocoded Shadow Layover Mask')
    parser.add_argument('--use_gpu', action='store_true', default=False, dest = 'use_gpu', help = 'Use GPU')


    return parser

def main():
    """
    main driver.
    """

    parser = create_parser()
    args = parser.parse_args()

    # init the SLC metadata reader
    slc = SLC(hdf5file=args.rslc)
    orbit = slc.getOrbit()

    # extract the radar grid parameters
    radargrid = slc.getRadarGrid(args.frequency)

    # extract orbit
    orbit = slc.getOrbit()

    # set defaults shared by both frequencies
    dem_raster = isce3.io.Raster(args.dem)
    epsg = dem_raster.get_epsg()
    proj = isce3.core.make_projection(epsg)
    ellipsoid = proj.ellipsoid

    # RSLC grid's Doppler (NISAR's RSLc grid is zero Doppler)
    grid_doppler = isce3.core.LUT2d()


    # compute shadow layover

    # construct Rdr2Geo
    # CPU or CUDA object 
    if args.use_gpu:
        Rdr2Geo = isce3.cuda.geometry.Rdr2Geo
    else:
        Rdr2Geo = isce3.geometry.Rdr2Geo

    rdr2geo_obj = Rdr2Geo(radargrid, orbit, ellipsoid, grid_doppler,
                              threshold=threshold, numiter=numiter,
                              extraiter=extraiter,
                              lines_per_block=lines_per_block)

    rdr2geo_scratch_path = os.path.join(os.path.abspath(os.path.dirname(args.output)), "scratch_rdrgeo")
    os.makedirs(rdr2geo_scratch_path, exist_ok=True)
    # run rdr2geo which also creates the shadow layover mask
    rdr2geo_obj.topo(dem_raster, rdr2geo_scratch_path)


    # geocode shadow layover

    # form the geogrid parameters
    #geo_grid = '' 
    ds = gdal.Open(f'NETCDF:"{args.gslc}"://science/LSAR/GSLC/grids/frequency{args.frequency}/{args.polarization}', gdal.GA_ReadOnly)
    geoTrans = ds.GetGeoTransform()
    width = ds.RasterXSize
    length = ds.RasterYSize

    proj = osr.SpatialReference(wkt=ds.GetProjection())
    epsg = proj.GetAttrValue('AUTHORITY',1)

    ds = None
    # the geocode object
    geo = isce3.geocode.GeocodeFloat32()
    geo.orbit = orbit
    geo.ellipsoid = ellipsoid
    geo.doppler = grid_doppler
    geo.threshold_geo2rdr = threshold_geo2rdr
    geo.numiter_geo2rdr = iteration_geo2rdr
    geo.dem_block_margin = dem_block_margin
    geo.lines_per_block = lines_per_block
    geo.data_interpolator = 'NEAREST'
    geo.geogrid(
                float(geoTrans[0]), #.start_x,
                float(geoTrans[3]), #.start_y,
                float(geoTrans[1]), #.spacing_x,
                float(geoTrans[5]), #.spacing_y,
                int(width),
                int(length),
                int(epsg),
            )

    # input is the shadow layover mask in radar coordinates
    input_raster_name = os.path.join(rdr2geo_scratch_path, "mask.rdr")
    input_raster = isce3.io.Raster(input_raster_name)

    geocoded_raster_name = os.path.abspath(args.output) 
    geocoded_raster = isce3.io.Raster(
                geocoded_raster_name, 
                width, length, 1,
                gdal.GDT_Float32, "ENVI")
    # run geocode
    geo.geocode(radar_grid=radargrid,
                input_raster=input_raster,
                output_raster=geocoded_raster,
                dem_raster=dem_raster,
                output_mode=isce3.geocode.GeocodeOutputMode.INTERP)
    #
     
if __name__ == '__main__':
    main()

