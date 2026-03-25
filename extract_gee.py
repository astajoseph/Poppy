//-----------------------------------------------
// 1. LOAD DATA
//-----------------------------------------------
var villages = ee.FeatureCollection(
  "projects/my-project-0001-487808/assets/MANDSAUR_V"
);

var excel = ee.FeatureCollection(
  "projects/my-project-0001-487808/assets/200"
);

//-----------------------------------------------
// 2. CLEAN & FILTER
//-----------------------------------------------
var excelClean = excel.map(function(f){
  return f.set('clean_name',
    ee.String(f.get('VILLAGENAME'))
      .toLowerCase()
      .replace(' ', '')
      .replace('-', '')
  );
});

var villageNames = excelClean.aggregate_array('clean_name');

var villagesClean = villages.map(function(f){
  return f.set('clean_name',
    ee.String(f.get('Villl_name'))
      .toLowerCase()
      .replace(' ', '')
      .replace('-', '')
  );
});

var selectedVillages = villagesClean.filter(
  ee.Filter.inList('clean_name', villageNames)
);

//-----------------------------------------------
// 3. SENTINEL-2 CLOUD MASK
//-----------------------------------------------
function maskS2(img){
  var qa = img.select('QA60');

  var cloud = 1 << 10;
  var cirrus = 1 << 11;

  var mask = qa.bitwiseAnd(cloud).eq(0)
      .and(qa.bitwiseAnd(cirrus).eq(0));

  return img.updateMask(mask).divide(10000);
}

//-----------------------------------------------
// 4. SENTINEL-2 DATA
//-----------------------------------------------
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterDate('2023-11-20','2023-12-05')
  .filterBounds(selectedVillages)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',20))
  .map(maskS2);

// Composite
var s2_image = s2.median();

// Bands
var B4 = s2_image.select('B4');   // RED
var B8 = s2_image.select('B8');   // NIR

var ndvi = s2_image.normalizedDifference(['B8','B4']).rename('NDVI');

//-----------------------------------------------
// 5. SENTINEL-1 (BACKSCATTER)
//-----------------------------------------------
var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterDate('2023-11-20','2023-12-05')
  .filterBounds(selectedVillages)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING')) // optional but stabilizes data
  .select(['VV','VH']);

// Composite
var s1_image = s1.median();

//-----------------------------------------------
// 6. FINAL STACK (ALL FEATURES)
//-----------------------------------------------
var finalImage = ee.Image.cat([
  B4.rename('RED'),
  B8.rename('NIR'),
  ndvi,
  s1_image.select('VV'),
  s1_image.select('VH')
]);

//-----------------------------------------------
// 7. DISPLAY
//-----------------------------------------------
Map.setCenter(75.37, 24.02, 12);

Map.addLayer(finalImage.select('NDVI'), 
             {min:0, max:1, palette:['white','green']}, 
             'NDVI');

Map.addLayer(finalImage.select('VV'), 
             {min:-20, max:0}, 
             'VV Backscatter');

Map.addLayer(finalImage.select('VH'), 
             {min:-25, max:-5}, 
             'VH Backscatter');

//-----------------------------------------------
// 8. EXTRACT VALUES PER VILLAGE
//-----------------------------------------------
var bandsImage = finalImage.select(['RED','NIR','NDVI','VV','VH']);

var villageStats = bandsImage.reduceRegions({
  collection: selectedVillages,
  reducer: ee.Reducer.mean(),
  scale: 10
});

// Keep only required fields
var villageStatsWithName = villageStats.map(function(f){
  return ee.Feature(null, {
    'Villl_name': f.get('Villl_name'),
    'RED': f.get('RED'),
    'NIR': f.get('NIR'),
    'NDVI': f.get('NDVI'),
    'VV': f.get('VV'),
    'VH': f.get('VH')
  });
});


//-----------------------------------------------
// 9. EXPORT TO CSV
//-----------------------------------------------
Export.table.toDrive({
  collection: villageStatsWithName,
  description: 'Village_All_Features_S2_S1',
  fileFormat: 'CSV'
});
