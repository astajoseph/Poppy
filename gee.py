//-----------------------------------------------
// 1. LOAD DATA
//-----------------------------------------------
var villages = ee.FeatureCollection(
  "projects/my-project-0001-487808/assets/Village"
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
// 3. CLOUD MASK
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
  .filterDate('2023-12-20','2024-01-05')
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

// FINAL STACK
var finalImage = ee.Image.cat([
  B4.rename('RED'),
  B8.rename('NIR'),
  ndvi
]);

//-----------------------------------------------
// 7. EXPORT ONE BIG IMAGE
//-----------------------------------------------
Export.image.toDrive({
  image: finalImage,
  description: 'Mandsaur_Full_Image2',
  folder: 'GEE_Master',
  fileNamePrefix: 'Mandsaur_Full',
  region: selectedVillages.geometry(),
  scale: 10,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});
