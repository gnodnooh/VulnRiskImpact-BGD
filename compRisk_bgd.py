#!/home/dlee/.virtualenvs/fh/bin/python
# -*- coding: utf-8 -*-
"""
This script assesses Flood-Health Vulnerability (FHV) and Risk 
author: Donghoon Lee / dlee298@wisc.edu
"""
import os
import numpy as np
import gdal
import fh
import pandas as pd

# Load Census indicators
cens = np.load('data_census.npy'); cens = cens.item()
hous, elec, watr, sani = cens['hous'], cens['elec'], cens['watr'], cens['sani']
resi, depd, disa, edul = cens['resi'], cens['depd'], cens['disa'], cens['edul']
empl, occu, litr = cens['empl'], cens['occu'], cens['litr']
ypop, opop, vpop = cens['ypop'], cens['opop'], cens['vpop']
# Load Other indicators
indc = np.load('data_indices.npy'); indc = indc.item()
priv, pcsh, ncsh, nhsp = indc['priv'], indc['pcsh'], indc['ncsh'], indc['nhsp']
nphc, slop, prec, tavg = indc['nphc'], indc['slop'], indc['prec'], indc['tavg']
wind, elev, wlth, povt = indc['wind'], indc['elev'], indc['wlth'], indc['povt']
incm, gdp, fpro, tphc = indc['incm'], indc['gdp'], indc['fpro'], indc['tphc']
aphc, thsp, ahsp, fdep = indc['aphc'], indc['thsp'], indc['ahsp'], indc['fdep']
# Load code4 and code3
fn = os.path.join('land', 'boundary_gadm', 'gadm4_code.tif')
ds = gdal.Open(fn); dsCopy = ds
code4 = ds.GetRasterBand(1).ReadAsArray()
code3 = np.floor(code4/100)
nc = (code4 == code4[0,0])      # No-code
yc = (code4 != code4[0,0])      # Yes-code


#%% Flood-Health Vulnerability (FHV)
#
# Susceptibility (100)
# - demo(110): ypop(111), opop(112), vpop(113), disa(114), depd(115)
# - seco(120): edul(121), litr(122), empl(123), occu(124), wlth(125), povt(126), incm(127)
# - heal(130): sani(131), watr(132)
# - hsys(140): nphc(141), nhsp(142), tphc(143), thsp(144), aphc(145), ahsp(146)
#
# Exposure (200)
# - phys(210): fpro(211), priv(212), elev(213), hous(214), resi(215)
# - cope(220): ncsh(221), pcsh(222), elec(223)
#
# Hazard(300)
# - hydr(310): fdep(311), slop(312)
# - clim(320): prec(321), tavg(322), wind(323)
#

# Matrix of all indicators
# - Both static and dynamic indicators (33)
dataYF = np.array([ypop[yc],opop[yc],vpop[yc],disa[yc],depd[yc],edul[yc],litr[yc],
                 empl[yc],occu[yc],wlth[yc],povt[yc],incm[yc],sani[yc],watr[yc],
                 nphc[yc],nhsp[yc],tphc[yc],thsp[yc],aphc[yc],ahsp[yc],fpro[yc],
                 priv[yc],elev[yc],hous[yc],resi[yc],ncsh[yc],pcsh[yc],elec[yc],
                 fdep[yc],slop[yc],prec[yc],tavg[yc],wind[yc]]).T
# - Only static indicators (30)
dataNF= np.array([ypop[yc],opop[yc],vpop[yc],disa[yc],depd[yc],edul[yc],litr[yc],
                 empl[yc],occu[yc],wlth[yc],povt[yc],incm[yc],sani[yc],watr[yc],
                 nphc[yc],nhsp[yc],tphc[yc],thsp[yc],fpro[yc],
                 priv[yc],elev[yc],hous[yc],resi[yc],ncsh[yc],pcsh[yc],elec[yc],
                 slop[yc],prec[yc],tavg[yc],wind[yc]]).T

# Load initial weights
fn = os.path.join('initial_weights.xlsx')
xl = pd.ExcelFile(fn); df = xl.parse('weight')
name = df.name
weightYF = df.weight
weightNF = df[~name.isin(['fdep','aphc','ahsp'])].weight
weightNF = weightNF/weightNF.sum()       # scale to 1

# Calculate FHV index
fhvYF = fh.valueToMap(np.dot(dataYF, weightYF), yc)
fhvNF = fh.valueToMap(np.dot(dataNF, weightNF), yc)

# Sub-domains composite indicators (rescaled to 0-1)
demo = np.dot(dataYF[:,:5], weightYF[:5])/weightYF[:5].sum()
demo = fh.valueToMap(demo, yc)
seco = np.dot(dataYF[:,5:12], weightYF[5:12])/weightYF[5:12].sum()
seco = fh.valueToMap(seco, yc)
heal = np.dot(dataYF[:,12:14], weightYF[12:14])/weightYF[12:14].sum()
heal = fh.valueToMap(heal, yc)
hsys = np.dot(dataYF[:,14:20], weightYF[14:20])/weightYF[14:20].sum()
hsys = fh.valueToMap(hsys, yc)
susc = np.dot(dataYF[:,:20], weightYF[:20])/weightYF[:20].sum()
susc = fh.valueToMap(susc, yc)
phys = np.dot(dataYF[:,20:25], weightYF[20:25])/weightYF[20:25].sum()
phys = fh.valueToMap(phys, yc)
cope = np.dot(dataYF[:,25:28], weightYF[25:28])/weightYF[25:28].sum()
cope = fh.valueToMap(cope, yc)
expo = np.dot(dataYF[:,20:28], weightYF[20:28])/weightYF[20:28].sum()
expo = fh.valueToMap(expo, yc)
hydr = np.dot(dataYF[:,28:30], weightYF[28:30])/weightYF[28:30].sum()
hydr = fh.valueToMap(hydr, yc)
clim = np.dot(dataYF[:,30:33], weightYF[30:33])/weightYF[30:33].sum()
clim = fh.valueToMap(clim, yc)
hazd = np.dot(dataYF[:,28:33], weightYF[28:33])/weightYF[28:33].sum()
hazd = fh.valueToMap(hazd, yc)

# Low and High FHV zone
hzon = np.ones(code4.shape)*99; hzon[(fhvYF >= 0.6)] = 1
lzon = np.ones(code4.shape)*99; lzon[(fhvYF < 0.4)] = 1

# Saving sub-domain composite indicators, FHV index, High FHV zone
fn = os.path.join('result','demo.tif'); temp = demo; temp[nc] = -9999
out_ds = fh.make_raster(dsCopy, fn, temp, gdal.GDT_Float32, -9999); del out_ds
fn = os.path.join('result','seco.tif'); temp = seco; temp[nc] = -9999
out_ds = fh.make_raster(dsCopy, fn, temp, gdal.GDT_Float32, -9999); del out_ds
fn = os.path.join('result','heal.tif'); temp = heal; temp[nc] = -9999
out_ds = fh.make_raster(dsCopy, fn, temp, gdal.GDT_Float32, -9999); del out_ds
fn = os.path.join('result','hsys.tif'); temp = hsys; temp[nc] = -9999
out_ds = fh.make_raster(dsCopy, fn, temp, gdal.GDT_Float32, -9999); del out_ds
fn = os.path.join('result','susc.tif'); temp = susc; temp[nc] = -9999
out_ds = fh.make_raster(dsCopy, fn, temp, gdal.GDT_Float32, -9999); del out_ds
fn = os.path.join('result','phys.tif'); temp = phys; temp[nc] = -9999
out_ds = fh.make_raster(dsCopy, fn, temp, gdal.GDT_Float32, -9999); del out_ds
fn = os.path.join('result','cope.tif'); temp = cope; temp[nc] = -9999
out_ds = fh.make_raster(dsCopy, fn, temp, gdal.GDT_Float32, -9999); del out_ds
fn = os.path.join('result','expo.tif'); temp = expo; temp[nc] = -9999
out_ds = fh.make_raster(dsCopy, fn, temp, gdal.GDT_Float32, -9999); del out_ds
fn = os.path.join('result','hydr.tif'); temp = hydr; temp[nc] = -9999
out_ds = fh.make_raster(dsCopy, fn, temp, gdal.GDT_Float32, -9999); del out_ds
fn = os.path.join('result','clim.tif'); temp = clim; temp[nc] = -9999
out_ds = fh.make_raster(dsCopy, fn, temp, gdal.GDT_Float32, -9999); del out_ds
fn = os.path.join('result','hazd.tif'); temp = hazd; temp[nc] = -9999
out_ds = fh.make_raster(dsCopy, fn, temp, gdal.GDT_Float32, -9999); del out_ds
fn = os.path.join('result','fhv.tif'); temp = fhvYF; temp[nc] = -9999
out_ds = fh.make_raster(dsCopy, fn, temp, gdal.GDT_Float32, -9999); del out_ds
fn = os.path.join('result','hzon.tif'); temp = hzon
out_ds = fh.make_raster(dsCopy, fn, temp, gdal.GDT_Int16, 99); del out_ds
fn = os.path.join('result','lzon.tif'); temp = lzon
out_ds = fh.make_raster(dsCopy, fn, temp, gdal.GDT_Int16, 99); del out_ds


#%% Flood and Health Risk Assessment
# Load flood depth
loc = os.path.join('hydrology', 'inundation_glofris')
ds = gdal.Open(os.path.join(loc, 'rp_00010.tif'))
depth = ds.GetRasterBand(1).ReadAsArray().astype('float')    # (727,559)
depth[nc] = 0
# Load population (LandScan, 2015)
fn = os.path.join('socioecon', 'population_landscan', 'lspop_bgd.tif')
ds = gdal.Open(fn)
popu = ds.GetRasterBand(1).ReadAsArray()
popu[(popu == popu.min()) | nc] = 0
# - 2015 population from World Bank
popu2015 = 161200886
# - Scale LandScan population to World Bank Population
popu = popu/popu.sum()*popu2015
popuTotl = popu.sum()
# =============================================================================
# # Load GDP(PPP) (Kummu et la., 2017)
# fn = os.path.join('socioecon', 'gdp_kummu', 'gdp_ppp_2015_30s_bgd.tif')
# ds = gdal.Open(fn)
# gdp = ds.GetRasterBand(1).ReadAsArray()
# gdp[np.isnan(gdp) | nc] = 0
# 
# =============================================================================
# - GDP/PPP MAP (World Bank)
# * GDP/PPP per capita 2015 (BGD): $3132.57 in 2011 constant international USD
gdp = popu.copy()
gdp = gdp * 3132.57


# 1. Flood Risk Assessment (FRA) -------------------------------------------- #
# (1) Affected Population
# *Affected population is assumed to increase linearly with water level until
#  2 meters.
thsd = 20
ratio = depth.copy()
ratio[ratio <= thsd] = ratio[ratio <= thsd]/thsd
ratio[ratio > thsd] = 1
popuAfft = np.sum(ratio*popu)

# (2) High-FHA and Low-FHA population
popuHzon = popu[fhvYF >= 0.60].sum()
popuLzon = popu[fhvYF < 0.40].sum()
popuMode = popu[(0.40<=fhvYF) & (fhvYF < 0.6)].sum()

# (3) Affected GDP
# *Affected GDP is assumed to increase linearly with water level from a damage
#  of zero for a water level of zero, to a maximum affected GDP at a water 
#  level of 3 meter.
thsd = 30
ratio = depth.copy()
ratio[ratio <= thsd] = ratio[ratio <= thsd]/thsd
ratio[ratio > thsd] = 1
gdpAfft = np.sum(ratio*gdp)


# 2. Health Risk Assessment (HRA) ------------------------------------------- #
# (1) Number of affected hospitals and PHC
import geopandas
thsd = 15
# - PHC
fn = os.path.join('health', 'healthsites_lged', 'family_rp00010.shp')
gdf = geopandas.read_file(fn)
phcDepth = gdf.rp_00010.values; nphc = len(phcDepth)
phcDepth = phcDepth[~np.isnan(phcDepth)]
phcDepth[phcDepth < thsd] = phcDepth[phcDepth < thsd]/thsd
phcDepth[phcDepth >= thsd] = 1
phcAfft = phcDepth.sum()
# - Hospital
fn = os.path.join('health', 'healthsites_lged', 'hospital_rp00010.shp')
gdf = geopandas.read_file(fn)
hspDepth = gdf.rp_00010.values; nhsp = len(hspDepth)
hspDepth = hspDepth[~np.isnan(hspDepth)]
hspDepth[hspDepth < thsd] = hspDepth[hspDepth < thsd]/thsd
hspDepth[hspDepth >= thsd] = 1
hspAfft = hspDepth.sum() 

# (2) Population with affected travel time to Hospitals and PHC
thsdAtt = 60
# - PHC
ds = gdal.Open(os.path.join('health', 'traveltime_lged', 'aphc.tif'))
phcAtt = ds.GetRasterBand(1).ReadAsArray().astype('float')
phcAtt[nc | (phcAtt < 0)]= 0
popuAttPhc = popu[phcAtt >= thsdAtt].sum()
# - Hospital
ds = gdal.Open(os.path.join('health', 'traveltime_lged', 'ahsp.tif'))
hspAtt = ds.GetRasterBand(1).ReadAsArray().astype('float')
hspAtt[nc | (hspAtt < 0)]= 0
popuAttHsp = popu[hspAtt >= thsdAtt].sum()

# Sensitivity analysis
fn = 'data_sensitivity'
ntrial = 5000                         # Number of trials
if not os.path.isfile(fn+'.npz'):
    
    freqYF = np.zeros([yc.sum(), 20]).astype('int16')
    freqNF = np.zeros([yc.sum(), 20]).astype('int16')
    for i in range(ntrial):
        # Generate random weights
        rweightYF = np.random.random([dataYF.shape[1], 1])
        rweightYF = rweightYF/rweightYF.sum()
        outputYF = np.dot(dataYF,rweightYF)
        rweightNF = np.random.random([dataNF.shape[1], 1])
        rweightNF = rweightNF/rweightNF.sum()
        outputNF = np.dot(dataNF,rweightNF)
        # Store frequencies
        for j in range(20):
            cid_yf, _ = np.where((j/20 <= outputYF) & (outputYF < (j+1)/20))
            freqYF[cid_yf,j] = freqYF[cid_yf,j] + 1
            cid_nf, _ = np.where((j/20 <= outputNF) & (outputNF < (j+1)/20))
            freqNF[cid_nf,j] = freqNF[cid_nf,j] + 1
        print('{}/{} ({:02.1f}%%)'.format(i+1,ntrial,(i+1)/ntrial*100))
    
    # Save result
    np.savez(fn, freqYF=freqYF, freqNF=freqNF)
    print('{}.npy is saved..'.format(fn))

else:
    # Load result
    output = np.load(fn+'.npz')
    freqYF, freqNF = output['freqYF'], output['freqNF']

# Frequency of Low-FHV and High-FHV
freqYFL = np.zeros(code4.shape)
freqYFL[yc] = freqYF[:,:8].sum(1)/ntrial
freqYFH = np.zeros(code4.shape)
freqYFH[yc] = freqYF[:,12:].sum(1)/ntrial
freqNFL = np.zeros(code4.shape)
freqNFL[yc] = freqNF[:,:8].sum(1)/ntrial
freqNFH = np.zeros(code4.shape)
freqNFH[yc] = freqNF[:,12:].sum(1)/ntrial

# Percentage of HZON outside or inside of predominantly vulnerable zone
thsd = 0.7
pdomHzonA = np.ones(code4.shape)*99; pdomHzonA[freqYFH >= thsd] = 1
pdomHzonB = np.ones(code4.shape)*99; pdomHzonB[freqNFH >= thsd] = 1
pdomHzonAreaA = np.sum(freqYFH >= thsd)/np.sum(yc)*100
pdomHzonAreaB = np.sum(freqNFH >= thsd)/np.sum(yc)*100
pdomHzonPopuA = np.sum(popu[freqYFH >= thsd])
pdomHzonPopuB = np.sum(popu[freqNFH >= thsd])
hzonInPdomA = np.sum((hzon == 1) & (freqYFH >= thsd))/np.sum(hzon == 1)
hzonInPdomB = np.sum((hzon == 1) & (freqNFH >= thsd))/np.sum(hzon == 1)

# Save maps
fn = os.path.join('result','freqYFL.tif'); temp = freqYFL; temp[nc] = -9999
out_ds = fh.make_raster(dsCopy, fn, temp, gdal.GDT_Float32, -9999); del out_ds
fn = os.path.join('result','freqYFH.tif'); temp = freqYFH; temp[nc] = -9999
out_ds = fh.make_raster(dsCopy, fn, temp, gdal.GDT_Float32, -9999); del out_ds
fn = os.path.join('result','freqNFL.tif'); temp = freqNFL; temp[nc] = -9999
out_ds = fh.make_raster(dsCopy, fn, temp, gdal.GDT_Float32, -9999); del out_ds
fn = os.path.join('result','freqNFH.tif'); temp = freqNFH; temp[nc] = -9999
out_ds = fh.make_raster(dsCopy, fn, temp, gdal.GDT_Float32, -9999); del out_ds
fn = os.path.join('result','pdomHzonA.tif'); temp = pdomHzonA
out_ds = fh.make_raster(dsCopy, fn, temp, gdal.GDT_Int16, 99); del out_ds
fn = os.path.join('result','pdomHzonB.tif'); temp = pdomHzonB
out_ds = fh.make_raster(dsCopy, fn, temp, gdal.GDT_Int16, 99); del out_ds



# Print results
print('==================================================')
print('VEH Layers and Risk')
print('--------------------------------------------------')
print('Susceptibility:\t\tMax= {:.3f}, Min= {:.3f}'.format(np.max(susc[yc]), np.min(susc[yc])))
print('- Demographic:\t\tMax= {:.3f}, Min= {:.3f}'.format(np.max(demo[yc]), np.min(demo[yc])))
print('- Socio-ecnominc:\tMax= {:.3f}, Min= {:.3f}'.format(np.max(seco[yc]), np.min(seco[yc])))
print('- Heath:\t\tMax= {:.3f}, Min= {:.3f}'.format(np.max(heal[yc]), np.min(heal[yc])))
print('- Health system:\tMax= {:.3f}, Min= {:.3f}'.format(np.max(hsys[yc]), np.min(hsys[yc])))
print('Exposure:\t\tMax= {:.3f}, Min= {:.3f}'.format(np.max(expo[yc]), np.min(expo[yc])))
print('- Physical exposure:\tMax= {:.3f}, Min= {:.3f}'.format(np.max(phys[yc]), np.min(phys[yc])))
print('- Coping capacity:\tMax= {:.3f}, Min= {:.3f}'.format(np.max(cope[yc]), np.min(cope[yc])))
print('Hazard:\t\t\tMax= {:.3f}, Min= {:.3f}'.format(np.max(hazd[yc]), np.min(hazd[yc])))
print('- Hydrologic:\t\tMax= {:.3f}, Min= {:.3f}'.format(np.max(hydr[yc]), np.min(hydr[yc])))
print('- Climatic:\t\tMax= {:.3f}, Min= {:.3f}'.format(np.max(clim[yc]), np.min(clim[yc])))
print('FHV:\t\t\tMax= {:.3f}, Min= {:.3f}'.format(np.max(fhvYF[yc]), np.min(fhvYF[yc])))
print('--------------------------------------------------')
print('Flood Risk Assessment')
print('--------------------------------------------------')
print('Affected population:\t{:>10,d} ({:.1f}%)'.format(int(popuAfft), popuAfft/popuTotl*100))
print('High-FHV population:\t{:>10,d} ({:.1f}%)'.format(int(popuHzon), popuHzon/popuTotl*100))
print('High-FHV area:\t\t{:.2f}%'.format(np.sum([fhvYF >= 0.6])/np.sum(yc)*100))
print('Low-FHV population:\t{:>10,d} ({:.1f}%)'.format(int(popuLzon), popuLzon/popuTotl*100))
print('Low-FHV area:\t\t{:.2f}%'.format(np.sum([(0<= fhvYF) & (fhvYF < 0.4) ])/np.sum(yc)*100))
print('Moderate-FHV population:{:>10,d} ({:.1f}%)'.format(int(popuMode), popuMode/popuTotl*100))
print('Moderate-FHV area:\t{:.2f}%'.format(np.sum([(0.4<= fhvYF) & (fhvYF < 0.6) ])/np.sum(yc)*100))
print('PdomHzon area-A:\t{:.2f}%'.format(pdomHzonAreaA))
print('PdomHzon area-B:\t{:.2f}%'.format(pdomHzonAreaB))
print('Popu in pdomHzon-A:\t{:>10,d} ({:.1f}%)'.format(int(pdomHzonPopuA), pdomHzonPopuA/popuTotl*100))
print('Popu in pdomHzon-B:\t{:>10,d} ({:.1f}%)'.format(int(pdomHzonPopuB), pdomHzonPopuB/popuTotl*100))
print('Hzon in pdomHzon-A:\t{:.2f}%'.format(hzonInPdomA*100))
print('Hzon in pdomHzon-B:\t{:.2f}%'.format(hzonInPdomB*100))
print('Affected GDP:\t\t${:>7,.1f} B'.format(gdpAfft/10**9))
print('Affected GDP per capita:${:>7,.1f} ({:.1f}%)'.format(gdpAfft/popuTotl, (gdpAfft/popuTotl)/(gdp.sum()/popuTotl)*100))
print('*GDP(PPP) per capita:\t${:>7,d}'.format(int(gdp.sum()/popuTotl)))
print('*Total population:     {:>,d}'.format(int(popuTotl)))
print('--------------------------------------------------')
print('Health Risk Assessment')
print('--------------------------------------------------')
print('Affected PHC:\t\t{:5d}/{:3d} ({:.1f}%)'.format(int(phcAfft), nphc, phcAfft/nphc*100))
print('Affected Hospitals:\t{:6d}/{:3d} ({:.1f}%)'.format(int(hspAfft), nhsp, hspAfft/nhsp*100))
print('ATT(>1hr) to PHC:\t{:>10,d} ({:.1f}%)'.format(int(popuAttPhc), popuAttPhc/popuTotl*100))
print('ATT(>1hr) to Hospital:\t{:>10,d} ({:.1f}%)'.format(int(popuAttHsp), popuAttHsp/popuTotl*100))
print('==================================================')
