# -*- coding: utf-8 -*-
"""
    This script includes core functions for pre-processing and vulnerability assessment.
    
    - ConvertShapeToRaster(shp_fn, rst_fn, out_fn, fieldname, out_dtype=rasterio.int32)
    - GenerateRaster(fn_out, meta, data, new_dtype=False, new_nodata=False)
    - ReprojectRaster(inpath, outpath, new_crs)
    - CropRasterShape(rst_fn, shp_fn, out_fn, all_touched=False)

    Revised at Apr-21-2020
    Donghoon Lee (dlee298@wisc.edu)
"""
import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio import transform
from rasterio.warp import calculate_default_transform, reproject, Resampling
import fiona
import gdal
import xlrd
import re
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# SPSS_PCA
from scipy.stats.mstats import zscore as ZSCORE
import mdp as MDP  # causing sklearn deprication warning
from operator import itemgetter
import copy

class SPSS_PCA:
    '''
    
    *** This code is adapted from https://github.com/geoss/sovi-validity ***
    
    A class that integrates most (all?) of the assumptions SPSS imbeds in their
    implimnetation of principle components analysis (PCA), which can be found in
    thier GUI under Analyze > Dimension Reduction > Factor. This class is not 
    intended to be a full blown recreation of the SPSS Factor Analysis GUI, but
    it does replicate (possibly) the most common use cases. Note that this class
    will not produce exactly the same results as SPSS, probably due to differences 
    in how eigenvectors/eigenvalues and/or singular values are computed. However, 
    this class does seem to get all the signs to match, which is not really necessary
    but kinda nice. Most of the approach came from the official SPSS documentation.

    References
    ----------
    ftp://public.dhe.ibm.com/software/analytics/spss/documentation/statistics/20.0/en/client/Manuals/IBM_SPSS_Statistics_Algorithms.pdf
    http://spssx-discussion.1045642.n5.nabble.com/Interpretation-of-PCA-td1074350.html
    http://mdp-toolkit.sourceforge.net/api/mdp.nodes.WhiteningNode-class.html
    https://github.com/mdp-toolkit/mdp-toolkit/blob/master/mdp/nodes/pca_nodes.py

    Parameters
    ----------
    inputs:  numpy array
             n x k numpy array; n observations and k variables on each observation
    reduce:  boolean (default=False)
             If True, then use eigenvalues to determine which factors to keep; all
             results will be based on just these factors. If False use all factors.
    min_eig: float (default=1.0)
             If reduce=True, then keep all factors with an eigenvalue greater than
             min_eig. SPSS default is 1.0. If reduce=False, then min_eig is ignored.
    varimax: boolean (default=False)
             If True, then apply a varimax rotation to the results. If False, then
             return the unrotated results only.

    Attributes
    ----------
    z_inputs:   numpy array
                z-scores of the input array.
    comp_mat:   numpy array
                Component matrix (a.k.a, "loadings").
    scores:     numpy array
                New uncorrelated vectors associated with each observation.
    eigenvals_all:  numpy array
                Eigenvalues associated with each factor.
    eigenvals:  numpy array
                Subset of eigenvalues_all reflecting only those that meet the 
                criterion defined by parameters reduce and min_eig.
    weights:    numpy array
                Values applied to the input data (after z-scores) to get the PCA
                scores. "Component score coefficient matrix" in SPSS or  
                "projection matrix" in the MDP library.
    comms:      numpy array
                Communalities
    sum_sq_load: numpy array
                 Sum of squared loadings.
    comp_mat_rot: numpy array or None
                  Component matrix after rotation. Ordered from highest to lowest 
                  variance explained based on sum_sq_load_rot. None if varimax=False.
    scores_rot: numpy array or None
                Uncorrelated vectors associated with each observation, after 
                rotation. None if varimax=False.
    weights_rot: numpy array or None
                Rotated values applied to the input data (after z-scores) to get 
                the PCA scores. None if varimax=False.
    sum_sq_load_rot: numpy array or None
                 Sum of squared loadings for rotated results. None if 
                 varimax=False.

    '''

    def __init__(self, inputs, reduce=False, min_eig=1.0, varimax=False):
        z_inputs = ZSCORE(inputs)  # seems necessary for SPSS "correlation matrix" setting (their default)

        # Run base SPSS-style PCA to get all eigenvalues
#         pca_node = MDP.nodes.PCANode()
        pca_node = MDP.nodes.WhiteningNode()  # Whitening mode
        scores = pca_node.execute(z_inputs)   # Base-run PCA 
        eigenvalues_all = pca_node.d          # Eigenvalues of the base-run

        # run SPSS-style PCA based on user settings
#         pca_node = MDP.nodes.PCANode(reduce=reduce, var_abs=min_eig)
        pca_node = MDP.nodes.WhiteningNode(reduce=reduce, var_abs=min_eig) # Retain only eigenvalue > min_eig
        scores = pca_node.execute(z_inputs)   # Principal components
        weights = pca_node.v                  # Transposed eigenvectors
        eigenvalues = pca_node.d              # Eigenvalues
        # *Component matrix (equal to correlation between X and PC)
        # *Component loadings = Eigenvector * np.sqrt(eigenvalue)
        # *The component loading is interpreted as the correlation of each item (x) with the PC.
        component_matrix = weights * eigenvalues  
        component_matrix = self._reflect(component_matrix)   # Get signs to match SPSS
        # *Summing the squared component loadings across the components (columns) 
        # *Communality is the variance in observed variables accounted for by a common factors. 
        communalities = (component_matrix**2).sum(1)         # Communalities (h**2, if full factors, 1)
        # Summing each squared loading down the items (rows) gives the eigenvalue.
        sum_sq_loadings = (component_matrix**2).sum(0)       # Same as eigenvalues 
        weights_reflected = component_matrix/eigenvalues     # Get signs to match SPSS
        scores_reflected = np.dot(z_inputs, weights_reflected)  # abs(scores)=abs(scores_reflected)

        if varimax:
            # SPSS-style varimax rotation prep (Scaling)
            c_normalizer = 1. / MDP.numx.sqrt(communalities)[:,None]    # Scaler
            cm_normalized = c_normalizer * component_matrix  # Normalized component matrix for varimax

            # Varimax rotation
            cm_normalized_varimax = self._varimax(cm_normalized)  # Run varimax
            c_normalizer2 = MDP.numx.sqrt(communalities)[:,None]
            cm_varimax = c_normalizer2 * cm_normalized_varimax    # Denormalized varimax output

            # Reorder varimax component matrix
            order = np.argsort(-np.sum(cm_varimax**2, 0))         # Sort by sum of squared loadings
            sum_sq_loadings_varimax = np.sum(cm_varimax**2, 0)[order]
            cm_varimax = cm_varimax[:,order]

            # Varimax scores
            cm_varimax_reflected = self._reflect(cm_varimax)  # Get signs to match SPSS
            varimax_weights = np.dot(cm_varimax_reflected, 
                              np.linalg.inv(np.dot(cm_varimax_reflected.T,
                              cm_varimax_reflected))) # CM(CM'CM)^-1
            scores_varimax = np.dot(z_inputs, varimax_weights)   # Principal components by Varimax
        else:
            comp_mat_rot = None
            scores_rot = None
            weights_rot = None
            cm_varimax_reflected = None
            scores_varimax = None
            varimax_weights = None
            sum_sq_loadings_varimax = None

        # Assign basic variables
        self.z_inputs = z_inputs
        self.scores = scores_reflected
        self.comp_mat = component_matrix
        self.eigenvals_all = eigenvalues_all
        self.eigenvals = eigenvalues
        self.weights = weights_reflected
        self.comms = communalities
        self.sum_sq_load = sum_sq_loadings
        # Assign Varimax variables
        self.comp_mat_rot = cm_varimax_reflected
        self.scores_rot = scores_varimax
        self.weights_rot = varimax_weights
        self.sum_sq_load_rot = sum_sq_loadings_varimax

    def _reflect(self, cm):
        # Reflect factors with negative sums; SPSS default
        cm = copy.deepcopy(cm)
        reflector = cm.sum(0)
        for column, measure in enumerate(reflector):
            if measure < 0:
                cm[:,column] = -cm[:,column]
        return cm

    def _varimax(self, Phi, gamma = 1.0, q = 100, tol = 1e-6):
        # downloaded from http://en.wikipedia.org/wiki/Talk%3aVarimax_rotation
        # also here http://stackoverflow.com/questions/17628589/perform-varimax-rotation-in-python-using-numpy
        p,k = Phi.shape
        R = np.eye(k)
        d=0
        for i in range(q):
            d_old = d
            Lambda = np.dot(Phi, R)
            u,s,vh = np.linalg.svd(np.dot(Phi.T,np.asarray(Lambda)**3 - (gamma/p) * 
                            np.dot(Lambda, np.diag(np.diag(np.dot(Lambda.T,Lambda))))))
            R = np.dot(u,vh)
            d = np.sum(s)
            if d_old!=0 and d/d_old < 1 + tol:
                break
        return np.dot(Phi, R)



# Colarmap and Colorbar controller
def cbarpam(bounds, color, labloc='on', boundaries=None, extension=None):
    '''Returns parameters for colormap and colorbar objects with a specified style.

        Parameters
        ----------
        bounds: list of bounds
        color: name of colormap or list of color names

        labloc: 'on' or 'in'
        boundaries: 
        extension: 'both', 'min', 'max'

        Return
        ------
        cmap: colormap
        norm: nomalization
        vmin: vmin for plotting
        vmax: vmax for plotting
        boundaries: boundaries for plotting
        
        Donghoon Lee @ Mar-15-2020
    '''
    
    gradient = np.linspace(0, 1, len(bounds)+1)
    # Create colorlist
    if type(color) is list:
        cmap = colors.ListedColormap(color,"")
    elif type(color) is str:
        cmap = plt.get_cmap(color, len(gradient))    
        # Extension
        colorsList = list(cmap(np.arange(len(gradient))))
        if extension is 'both':
            cmap = colors.ListedColormap(colorsList[1:-1],"")
            cmap.set_under(colorsList[0])
            cmap.set_over(colorsList[-1])
        elif extension is 'max':
            cmap = colors.ListedColormap(colorsList[:-1],"")
            cmap.set_over(colorsList[-1])
        elif extension is 'min':
            cmap = colors.ListedColormap(colorsList[1:],"")
            cmap.set_under(colorsList[0])
        elif extension is None:
            gradient = np.linspace(0, 1, len(bounds)-1)
            cmap = plt.get_cmap(color, len(gradient))
        else:
            raise ValueError('Check the extension')
    else:
        raise ValueError('Check the type of color.')
    # Normalization
    norm = colors.BoundaryNorm(bounds, cmap.N)
    # vmin and vmax
    vmin=bounds[0]
    vmax=bounds[-1]
    # Ticks
    if labloc == 'on':
        ticks = bounds
    elif labloc == 'in':
        ticks = np.array(bounds)[0:-1] + (np.array(bounds)[1:] - np.array(bounds)[0:-1])/2
    
    return cmap, norm, vmin, vmax, ticks, boundaries


def ConvertShapeToRaster(shp_fn, rst_fn, out_fn, fieldname, out_dtype=rasterio.int32):
    """Convert shapefile to a raster with reference raster
    """
    # Open the shapefile with GeoPandas
    unit = gpd.read_file(shp_fn)
    # Open the raster file as a template for feature burning using rasterio
    rst = rasterio.open(rst_fn)
    # Copy and update the metadata frm the input raster for the output
    profile = rst.profile.copy()
    profile.update(
        dtype=out_dtype,
        compress='lzw')
    # Before burning it, we need to 
    unit = unit.assign(ID_int = unit[fieldname].values.astype(out_dtype))
    # Burn the features into the raster and write it out
    with rasterio.open(out_fn, 'w+', **profile) as out:
        out_arr = out.read(1)
        shapes = ((geom, value) for geom, value in zip(unit.geometry, unit.ID_int))
        burned = rasterio.features.rasterize(shapes=shapes, fill=0, out=out_arr, 
                                             transform=out.transform,
                                             all_touched=False)
        out.write_band(1, burned)
    print('%s is saved' % out_fn)


def ValidCellToMap(data, valid, dtype='float32', nodata=-9999):
    """Convert values of valid cells to 2d Ndarray map format.
    """

    assert valid.sum() == data.shape[0]
    tmap = np.ones(valid.shape)*nodata
    tmap[valid] = data
    return tmap.astype(dtype)


def GenerateRaster(fn_out, meta, data, new_dtype=False, new_nodata=False):

    # New Dtype
    if new_dtype is not False:
        meta.update({'dtype': new_dtype})
    # New Nodata value
    if new_nodata is not False:
        meta.update({'nodata': new_nodata})
    # Write a raster
    with rasterio.open(fn_out, 'w+', **meta) as dst:
        dst.write_band(1, data)
        print('%s is saved.' % fn_out)



def ReprojectRaster(inpath, outpath, new_crs):
    """Reproject a raster with a specific crs
    """
    dst_crs = new_crs # CRS for web meractor 

    with rasterio.open(inpath) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(outpath, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
            print("%s is saved." % outpath)

    
    
def ReprojectToReference(in_path, ref_path, out_path, out_dtype, out_nodata=None):
    """Reproject a raster to reference raster
    """
    with rasterio.open(ref_path) as ref:
        profile = ref.profile.copy()
        profile.update(
            dtype=out_dtype,
            nodata=out_nodata,
            compress='lzw')
        with rasterio.open(in_path) as src:
            with rasterio.open(out_path, 'w', **profile) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst.transform,
                        dst_crs=dst.crs,
                        resampling=Resampling.nearest)

                print("%s is saved." % out_path)
    
    

def CropRasterShape(rst_fn, shp_fn, out_fn, all_touched=False):
    """crops raster with shapefile and save as new raster (GeoTiff)


    """
    # Get feature of the polygon (supposed to be a single polygon)
    with fiona.open(shp_fn, 'r') as shapefile:
        geoms = [feature['geometry'] for feature in shapefile]
    # Crop raster including cells over the lines (all_touched)
    with rasterio.open(rst_fn) as src:
        out_image, out_transform = mask(src, geoms, 
                                        crop=True, 
                                        all_touched=all_touched)
        out_meta = src.meta.copy()
    # Update spatial transform and height & width
    out_meta.update({'driver': 'GTiff',
                     'height': out_image.shape[1],
                     'width': out_image.shape[2],
                     'transform': out_transform})
    # Write the cropped raster
    with rasterio.open(out_fn, 'w', **out_meta) as dest:
        dest.write(out_image)
        print('%s is saved.' % out_fn)


def LoadCensusINEI(fn_census, fn_label):
    '''
    Read INEI 2017 National Census data (Excel) as Pandas dataframe format.
    Spanish labels are replaced by English labels.
    '''
# =============================================================================
#     #%% INPUT
#     fn = os.path.join('census', 'P08AFILIA.xlsx')
# =============================================================================
    
    # Read variable from Excel file
    df = pd.read_excel(fn_census, 
                       skiprows = 5,
                       header=0, 
                       index_col=1,
                       skipfooter=3)
    df = df.loc[:, ~df.columns.str.contains('^unnamed', case=False)]
    cols = df.columns
    
    # Update Spanish labels to English
    dfLabel = pd.read_excel(fn_label)
    # Find all rows of variable code and value
    rowCode = np.squeeze(np.where(dfLabel['Spanish'].str.match('Nombre :') == True))
    rowLabel = np.squeeze(np.where(dfLabel['Spanish'].str.match('Value Labels') == True))
    assert len(rowCode) == len(rowLabel)
    # Find a row of the target code
    code = re.split(r"/|.xlsx", fn_census)[-2]
    cond1 = dfLabel['Spanish'].str.match('Nombre : %s' % code)
    cond2 = dfLabel['Spanish'].str.len() == (9 + len(code))
    row = np.where(cond1 & cond2)[0][0]
    # Read both Spanish and English labels
    idx = rowCode.searchsorted(row)
    df2 = dfLabel.iloc[rowLabel[idx]+1:rowCode[idx+1]]
    label_spn = df2['Spanish'].apply(lambda x: x[x.find('. ')+2:])
    label_eng = df2['English'].apply(lambda x: x[x.find('. ')+2:])
    # Check the number of columns
    nlabel = len(label_spn)
    assert nlabel == np.in1d(cols, label_spn).sum()
    # Replace Spanish labels to English
    index = [np.where(label_spn == x)[0][0] for i, x in enumerate(cols[1:])]
    df.columns = ['District'] + list(label_eng.values[index])
    df.index.name='IDDIST'
    return df

    
def CorrectDistrict(dfCensus, method):
    # District map is not consistent with 2017 Census's districts.
    # 120604 (Mazamari) and 120606 (Pangoa) of census data are merged to
    # 120699 (MAZAMARI - PANGOA) of district map
    idMerg = [120604, 120606]    
    df = dfCensus.copy()
    if method == 'sum':
        df.loc[120699] = df.loc[idMerg].sum()
    elif method == 'average':
        df.loc[120699] = df.loc[idMerg].mean()
    elif method == 'min':
        df.loc[120699] = df.loc[idMerg].min()
    elif method == 'max':
        df.loc[120699] = df.loc[idMerg].max()
    df.loc[120699].District = 'Junín, Satipo, distrito de Mazamari-Pagoa'
    return df.drop(idMerg)


def TTimeCategory(array):
    '''
    Scale travel time to 1-8
    '''
    time = array[~np.isnan(array)]
    time[time < 30] = 0
    time[(30 <= time) & (time < 60)] = 1
    time[(60 <= time) & (time < 120)] = 2
    time[(120 <= time) & (time < 180)] = 3
    time[(180 <= time) & (time < 360)] = 4
    time[(360 <= time) & (time < 720)] = 5
    time[(720 <= time) & (time < 1440)] = 6
    time[(1440 <= time) & (time < 3000)] = 7
    time[time >= 3000] = 8
    array[~np.isnan(array)] = time
    return array


def censusToRaster(out_fn, meta, idmap, data):

# =============================================================================
#     #%% Input
#     out_fn = './census/test.tif'
#     idmap = did.copy()
#     data = page5.copy()
# =============================================================================
    
    # Change metadata
    meta['dtype'] = rasterio.float32
    idmap = idmap.astype(rasterio.float32)
    meta['nodata'] = -9999
    idmap[idmap == idmap[0,0]] = -9999

    # Compare IDs between census Dataframe and idMap
    listImap = np.unique(idmap[idmap != idmap[0,0]])
    listData = data.index.values
    assert len(listImap) == len(listData)
    
    # Distributes data    
    for i in listData:
        idmap[idmap == i] = data[i]
    
    # Write a raster
    with rasterio.open(out_fn, 'w', **meta) as dest:
        dest.write(idmap[None,:,:])
        print('%s is saved.' % out_fn)
        
        



    
    
def zeroToOne(array):
    '''
    Scale data from 0 to 1
    '''
    data = array[~np.isnan(array)]
    data = (data - data.min())/(data.max()-data.min())
    array[~np.isnan(array)] = data
    
    return array


def affectedGdpFlood(gdp, fdep):
    '''
    Calculated total affected GDP by flood levels
    '''
    
    gdpdata = gdp.copy(); depth = fdep.copy()
    
    depth[depth <= 30] = depth[depth <= 30]/30
    depth[depth > 30] = 1
    
    gdpAfft = np.sum(depth*gdpdata)
    
    return gdpAfft
    


##### CODE FOR BANGLADESH WORK #####
def LoadCensusBBS(fn_census):
    '''
    Read BBS 2011 National Census data (Excel) as Pandas dataframe format.
    '''
    remove = ['Unnamed: 0','Upazila/Thana Name']
    df = pd.read_excel(fn_census,
                       skiprows=9,
                       header=0,
                       index_col=0,
                       skipfooter=4,
                       usecols=lambda x: x not in remove)
    
    return df


def LoadUnionAge5(fn):
    '''
    Reads BBS 2011 Census - Union Age5 population data (e.g., age5_Rangpur.xls)
    
    Parameters
    ----------
    fn: file path

    Return
    ------
    df: dataframe
    
    Donghoon @ Mar-22-2019
    '''
    df = pd.read_excel(fn,               
                       skiprows=11,
                       header=[0],
                       index_col=[0,1],
                       skipfooter=4)
    df = df.droplevel(level=0)
    df.columns = ['Age5', 'Male', 'Female', 'Total']
    # Remove Union Total rows
    df = df[~((df['Male'] == 'Male') & (df['Female'] == 'Female') & (df['Total'] == 'Total'))]
    # Set MultiIndex
    df.reset_index(inplace=True)
    df = df.set_index(['Union','Age5'])
    # Rename Age 5 years group
    idxLv1 = ['00-04','05-09','10-14','15-19','20-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69','70-74','75-79','80+','Total']
    df.index = df.index.set_levels(idxLv1, level=1)
    # Drop the division 'Total' rows
    df = df.drop('Total',level=0)
    # Reset index
    df = df.reset_index()
    return df


def make_raster(in_ds, fn, data, data_type, nodata=None):
    """Create a one-band GeoTiff.

    in_ds     - datasource to copy projection and geotransform from
    fn        - path to the file to create
    data      - Numpy array containing data to archive
    data_type - output data type
    nodata    - optional NoData burn_values
    """

    driver = gdal.GetDriverByName('gtiff')
    out_ds = driver.Create(
        fn, in_ds.RasterXSize, in_ds.RasterYSize, 1, data_type)
    out_ds.SetProjection(in_ds.GetProjection())
    out_ds.SetGeoTransform(in_ds.GetGeoTransform())
    out_band = out_ds.GetRasterBand(1)
    if nodata is not None:
        out_band.SetNoDataValue(nodata)
    out_band.WriteArray(data)
    out_band.FlushCache()
    #out_band.ComputerStaitstics(False)
    print('"{}" is printed.'.format(fn))
    return out_ds


def upazilaToTable(df, noi, column):
    """Extracts Upazila level vertical data
    """
    import numpy as np
    
#    noi = ('Male', 'Female'); column = 'B'
    codeUp = df.A.str.extract('(\d+)')
    codeUp = codeUp[~codeUp.isna().values].values.astype(int)
    ioi = df.A.str.startswith(noi, na=False)
    count = df.loc[ioi, column].values
    count = count.reshape([int(len(count)/len(noi)),len(noi)]).astype(int)
    count = count[:-1:,:]
    table = np.concatenate((codeUp, count), 1)
    
    return table
    

def valueToMap(value, code):
    '''
    Distribute value to Yes-Code region
    '''
    
    output = np.zeros(code.shape)
    output[code] = value
    
    return output


def evaluation(name, index, code4):
    
#    index = hous
    mask = (code4 == code4[0,0])
    core = index[~mask]
    print('{} max: {:.3f}, min: {:.3f}'.format(name, core.max(), core.min()))
    
def climInterpolate(clim, code4):
    
#    clim = prec.copy()
    
    x,y = np.where((clim == clim.min()) & (code4 != code4[0,0]))
    clim[x,y] = clim[clim != clim.min()].mean()
        
    return clim




#%%
# =============================================================================
# def censusToRaster(ds, fn, imap, data):
#     '''
#     Distributes district-level data to spatial map
#     '''
#     import os
#     imap = imap.copy()
#     
#     # Distributes data    
#     for i in range(len(data)):
#         imap[imap == data[i,0]] = data[i,1]
# 
#     # Save new raster
#     if not os.path.isfile(fn):
#         out_ds = make_raster(ds, fn, imap, gdal.GDT_Float64, imap[0,0])
#         del out_ds
# 
#     return imap    
# =============================================================================