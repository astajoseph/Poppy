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
// 3. CLOUD MASK FUNCTION
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
// 4. DATASET
//-----------------------------------------------
var dataset = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterDate('2023-11-20','2023-12-05')
  .filterBounds(selectedVillages)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',20))
  .map(maskS2);

//-----------------------------------------------
// 5. COMPOSITE
//-----------------------------------------------
var image = dataset.median();

//-----------------------------------------------
// 6. BANDS + NDVI
//-----------------------------------------------
var B4 = image.select('B4');   // RED
var B8 = image.select('B8');   // NIR

var ndvi = image.normalizedDifference(['B8','B4']).rename('NDVI');

var finalImage = ee.Image.cat([
  B4.rename('RED'),
  B8.rename('NIR'),
  ndvi
]);

//-----------------------------------------------
// 7. DISPLAY NDVI
//-----------------------------------------------
Map.setCenter(75.37, 24.02, 12); // Center on Mandsaur
Map.addLayer(finalImage.select('NDVI'), 
             {min:0, max:1, palette:['white','green']}, 
             'NDVI');

//-----------------------------------------------
// 8. EXTRACT RED, NIR & NDVI PER VILLAGE
//-----------------------------------------------
var bandsImage = finalImage.select(['RED','NIR','NDVI']);

var villageStats = bandsImage.reduceRegions({
  collection: selectedVillages,
  reducer: ee.Reducer.mean(),
  scale: 10
});

var villageStatsWithName = villageStats.map(function(f){
  return ee.Feature(null, {
    'Villl_name': f.get('Villl_name'),
    'RED': f.get('RED'),
    'NIR': f.get('NIR'),
    'NDVI': f.get('NDVI')
  });
});

print('Village RED, NIR, NDVI', villageStatsWithName);

//-----------------------------------------------
// 9. EXPORT TO CSV
//-----------------------------------------------
Export.table.toDrive({
  collection: villageStatsWithName,
  description: 'Village_Red_NIR_NDVI',
  fileFormat: 'CSV'
});
