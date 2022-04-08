#!/home/dlee/.virtualenvs/fh/bin/python
# -*- coding: utf-8 -*-
'''
This script imports BBS census 2011 data and distribute to 1km x 1km map data.
The original census data was obtained from:
http://203.112.218.69/binbgd/RpWebEngine.exe/Portal
'''
import os
import numpy as np
import gdal
import fh
import pandas as pd


#%% Load GADM4 raster
fn_adm = os.path.join('land', 'boundary_gadm', 'gadm4.tif')
ds = gdal.Open(fn_adm)
code4 = ds.GetRasterBand(1).ReadAsArray().astype('uint32')
# - Load Value-and-CC4
xl = pd.ExcelFile(os.path.join('land', 'boundary_gadm', 'gadm4.xls'))
df = xl.parse('gadm4.tif.vat')
table = np.array([df.Value, df.CC_4]).T
# - Change code4 map to CC4 values
code4[code4 == code4.max()] = 4294967295
for i in range(len(table)):
    code4[code4 == table[i,0]] = table[i,1]
code3 = np.floor(code4/100)
# - Save new raster
fn = os.path.join('land', 'boundary_gadm', 'gadm4_code.tif')
if not os.path.isfile(fn):
    out_ds = fh.make_raster(ds, fn, code4, gdal.GDT_UInt32, 4294967295)
    del out_ds


#%% Load BBS Census 2011 data
loc = './socioecon/census_2011/'

# Household Variables
'''Obtained "Household" variables:
- Type of House
- Type of Household
- Number of House
- Tenancy of House
- Source of Drinking Water
- Toilet Facilities
- Electricity Connection
- RMO
- Area of Residence
'''
# - Type of House (hous, 0-1) 
# (#house_Kutcha_and_Jhupri / #house_total)
# *Pucca means high quality materials (e.g., cement or RCC)
# *Kutcha & Jhupri means weaker materials (e.g., mud, clay, lime, or thatched)
fn = os.path.join(loc, 'type_of_house.xls')
code, df = fh.bbsCensus(fn)
columns = df.columns
hous = df[['Kutcha', 'Jhupri']].sum(1)/df.sum(1)
hous = np.concatenate((code, hous.values[:,None]), axis=1)
# - Scale to 0-1
hous[:,1] = fh.zeroToOne(hous[:,1])
hous = fh.censusToRaster(ds, os.path.join(loc,'raster','hous.tif'), code3, hous)
fh.evaluation('hous', hous, code4)

# - Electricity connection (elec, 0-1)
# (#house_without_electricity / #house_total)
fn = os.path.join(loc, 'electricity_connection.xls')
code, df = fh.bbsCensus(fn)
elec = df['No']/df[['Yes','No']].sum(1)
elec = np.concatenate((code, elec.values[:,None]), axis=1)
# - Scale to 0-1
elec[:,1] = fh.zeroToOne(elec[:,1])
elec = fh.censusToRaster(ds, os.path.join(loc,'raster','elec.tif'), code3, elec)
fh.evaluation('elec', elec, code4)

# - Source of drinking water (watr, 0-1)
# (#house_with_Tap_and_Tube-well' / #house_total)
fn = os.path.join(loc, 'source_of_drinking_water.xls')
code, df = fh.bbsCensus(fn)
watr = df['Other']/df.sum(1)
watr = np.concatenate((code, watr.values[:,None]), axis=1)
# - Scale to 0-1
watr[:,1] = fh.zeroToOne(watr[:,1])
watr = fh.censusToRaster(ds, os.path.join(loc,'raster','watr.tif'), code3, watr)
fh.evaluation('watr', watr, code4)

# - Toilet facility (sani, 0-1)
# (#house_without_sanitary / #house_total)
fn = os.path.join(loc, 'toilet_facilities.xls')
code, df = fh.bbsCensus(fn)
columns = df.columns
sani = df[['Non-Sanitary', 'None']].sum(1)/df.sum(1)
sani = np.concatenate((code, sani.values[:,None]), axis=1)
# - Scale to 0-1
sani[:,1] = fh.zeroToOne(sani[:,1])
sani = fh.censusToRaster(ds, os.path.join(loc,'raster','sani.tif'), code3, sani)
fh.evaluation('sani', sani, code4)

# - Area of Residence (resi, 0-1)
# (#house_in_rural / #house_total)
fn = os.path.join(loc, 'area_of_residence.xls')
code, df = fh.bbsCensus(fn)
columns = df.columns
resi = df['Rural']/df.sum(1)
resi = np.concatenate((code, resi.values[:,None]), axis=1)
# - Scale to 0-1
resi[:,1] = fh.zeroToOne(resi[:,1])
resi = fh.censusToRaster(ds, os.path.join(loc,'raster','resi.tif'), code3, resi)
fh.evaluation('resi', resi, code4)

# Population Variables
'''Obtained "Population" variables:
- Age 5 years group
- Sex
- Disability
- Student
- Educational Attainment
- Field of Education
- Literacy
- Activity Status
- Employment Field
'''
# - Age group in 5 years
# pop of each 5 years age group
fn = os.path.join(loc, 'age_group_5.xls')
code, df = fh.bbsCensus(fn)
columns = list(df.columns)
columns[0:3] = ['0 - 4', '5 - 9', '10 - 14']
df.columns = columns
age5 = df.values.astype(int)
pop = np.concatenate((code, df.sum(1).values[:,None]), axis=1)

# - Population of Young (ypop, 0-1)
# (#pop_aged_0-14 / #pop_total)
ypop = age5[:,0:3].sum(1)/pop[:,1]
ypop = np.concatenate((code, ypop[:,None]),axis=1)
# - Scale to 0-1 
ypop[:,1] = fh.zeroToOne(ypop[:,1])
ypop = fh.censusToRaster(ds, os.path.join(loc,'raster','ypop.tif'), code3, ypop)
fh.evaluation('ypop', ypop, code4)

# - Population of Old (opop, 0-1)
# (#pop_aged_65+ / #pop_total)
opop = age5[:,-4:].sum(1)/pop[:,1]
opop = np.concatenate((code, opop[:,None]),axis=1)
# - Scale to 0-1
opop[:,1] = fh.zeroToOne(opop[:,1])
opop = fh.censusToRaster(ds, os.path.join(loc,'raster','opop.tif'), code3, opop)
fh.evaluation('opop', opop, code4)

# - Dependency Ratio (depd, 0-1)
# (#pop_aged_0-14_and_65-80+ / #pop_aged 15-64)
depd = np.sum(age5[:,[0,1,2,11,12,13,14]],1)/np.sum(age5[:,3:11],1)
depd = np.concatenate((code, depd[:,None]),axis=1)
# - Scale to 0-1
depd[:,1] = fh.zeroToOne(depd[:,1])
depd = fh.censusToRaster(ds, os.path.join(loc,'raster','depd.tif'), code3, depd)
fh.evaluation('depd', depd, code4)

# - Sex
# *pop of male and female
fn = os.path.join(loc, 'sex.xls')
code, df = fh.bbsCensus(fn)
sex = df.values.astype(int)
sex = np.concatenate((code, sex), axis=1)

# - Vulnerable people (pvul, 0-1)
# (#female_pop_aged_0-14_and_65+ / #pop_total)
sr = sex[:,2]/sex[:,1:3].sum(1)
vpop = (age5[:,0:3].sum(1) + age5[:,-4:].sum(1))*sr / pop[:,1]
vpop = np.concatenate((code, vpop[:,None]),axis=1)
# - Scale to 0-1
vpop[:,1] = fh.zeroToOne(vpop[:,1])
vpop = fh.censusToRaster(ds, os.path.join(loc,'raster','vpop.tif'), code3, vpop)
fh.evaluation('vpop', vpop, code4)

# - Disability (disa, 0-1)
# (#pop_with_disability / #pop_total)
fn = os.path.join(loc, 'disability.xls')
code, df = fh.bbsCensus(fn)
columns = df.columns
disa = df[columns[1:]].sum(1)/df.sum(1)
disa = np.concatenate((code, disa.values[:,None]), axis=1)
# - Scale to 0-1
disa[:,1] = fh.zeroToOne(disa[:,1])
disa = fh.censusToRaster(ds, os.path.join(loc,'raster','disa.tif'), code3, disa)
fh.evaluation('disa', disa, code4)

# - Education level (edul, 0-1)
# (#pop_without_primary_education / #pop_total)
# *BGD's primary education is ClassI-ClassV
# (https://en.wikipedia.org/wiki/Education_in_Bangladesh#/media/File:BangEduSys.png)
fn = os.path.join(loc, 'education_attainment.xls')
code, df = fh.bbsCensus(fn)
columns = df.columns
edul = df[columns[:5]].sum(1) / df.sum(1)
edul = np.concatenate((code, edul.values[:,None]), axis=1)
# - Scale to 0-1
edul[:,1] = fh.zeroToOne(edul[:,1])
edul = fh.censusToRaster(ds, os.path.join(loc,'raster','edul.tif'), code3, edul)
fh.evaluation('edul', edul, code4)

# - Employment (empl, 0-1)
# (#pop_without_employment / #pop_total)
fn = os.path.join(loc, 'activity_status.xls')
code, df = fh.bbsCensus(fn)
columns = df.columns
empl = df[columns[[1,3]]].sum(1)/df.sum(1)
empl = np.concatenate((code, empl.values[:,None]), axis=1)
# - Scale to 0-1
empl[:,1] = fh.zeroToOne(empl[:,1])
empl = fh.censusToRaster(ds, os.path.join(loc,'raster','empl.tif'), code3, empl)
fh.evaluation('empl', empl, code4)

# - Employment field (occu, 0-1)
# (#pop_of_Industry_and_Service / #pop_total)
fn = os.path.join(loc, 'employment_field.xls')
code, df = fh.bbsCensus(fn)
columns = df.columns
occu = df['Agriculture']/df.sum(1)
occu = np.concatenate((code, occu.values[:,None]), axis=1)
# - Scale to 0-1
occu[:,1] = fh.zeroToOne(occu[:,1])
occu = fh.censusToRaster(ds, os.path.join(loc,'raster','occu.tif'), code3, occu)
fh.evaluation('occu', occu, code4)

# - Literacy (litr, 0-1)
# (#pop without literacy / #pop_total)
fn = os.path.join(loc, 'literacy.xls')
code, df = fh.bbsCensus(fn)
litr = df['No']/df.sum(1)
litr = np.concatenate((code, litr.values[:,None]), axis=1)
# - Scale to 0-1
litr[:,1] = fh.zeroToOne(litr[:,1])
litr = fh.censusToRaster(ds, os.path.join(loc,'raster','litr.tif'), code3, litr)
fh.evaluation('litr', litr, code4)

# Save files
fn = 'data_census'
data = {'hous':hous,'elec':elec,'watr':watr,'sani':sani,'resi':resi,'depd':depd,
       'disa':disa,'edul':edul,'empl':empl,'occu':occu,'litr':litr,
       'pop':pop,'ypop':ypop,'opop':opop,'vpop':vpop}
np.save(fn, data)
print('{}.npy is saved..'.format(fn))










