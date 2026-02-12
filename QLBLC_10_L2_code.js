var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED');
var s2_cloud = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY');

var point = table;
Map.centerObject(point, 8);

var seasonalPeriods = [
  {name: 'winter', start: '2024-02-01', end: '2024-03-31'},
  {name: 'spring', start: '2024-04-01', end: '2024-05-31'},
  {name: 'summer', start: '2024-07-01', end: '2024-08-31'},
  {name: 'autumn', start: '2024-10-01', end: '2024-11-30'}
];

function rmCloudByProbability(image, threshold) {
    return image.updateMask(image.select('probability').lte(threshold));
}
function scaleImage(image) {
    return image.divide(10000).set('system:time_start', image.get('system:time_start'));
}
function getMergeImages(primary, secondary) {
    return ee.ImageCollection(ee.Join.inner().apply(primary, secondary, 
        ee.Filter.equals({ leftField: 'system:index', rightField: 'system:index' })
    )).map(function(image) {
        return ee.Image(image.get('primary')).addBands(image.get('secondary'));
    });
}
function fillMissingValues(image, radius, iterations) {
    return image.unmask(image.focal_mean({
        kernel: ee.Kernel.square({ radius: radius, units: 'pixels' }),
        iterations: iterations
    }));
}

var dem_10m = ee.ImageCollection("COPERNICUS/DEM/GLO30")
             .filterBounds(qhh_border)
             .map(function(image) {
               return image.select('DEM')
                           .clip(qhh_border)
                           .resample('bilinear')
                           .reproject({crs: 'EPSG:4326', scale: 10});
             })
             .mosaic();

var demFeatures = ee.Image.constant(1).rename('constant')
    .addBands([
      dem_10m.lt(3700).rename('Below_3700m'),
      dem_10m.gte(3800).rename('Above_3800m'),
      dem_10m.gte(3200).rename('Above_3200m')
  ]);

function extractSeasonalFeatures(period) {
    var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterDate(period.start, period.end)
        .filterBounds(point)
        .map(scaleImage);
    var s2_cloud = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        .filterDate(period.start, period.end)
        .filterBounds(point);
    var merged = getMergeImages(s2, s2_cloud).map(function(image) {
        return rmCloudByProbability(image, 20);
    });
    var medianS2 = merged.median().clip(point).toFloat();
    var filled = fillMissingValues(medianS2, 2, 3);
    var clipped = filled.clipToCollection(qhh_border);

    var bandNames = ['B2','B3','B4','B5','B6','B7','B8', 'B11'];
    var renamedBands = bandNames.map(function(band) {
        return band + '_' + period.name;
    });
    clipped = clipped.select(bandNames).rename(renamedBands);

    var indices = {
        NDVI: clipped.normalizedDifference(['B8_' + period.name, 'B4_' + period.name]).rename('NDVI_' + period.name),
        NDBI: clipped.normalizedDifference(['B11_' + period.name, 'B8_' + period.name]).rename('NDBI_' + period.name),  
        MNDWI: clipped.normalizedDifference(['B3_' + period.name, 'B11_' + period.name]).rename('MNDWI_' + period.name),
        BSI: clipped.normalizedDifference(['B11_' + period.name, 'B4_' + period.name]).rename('BSI_' + period.name)
    };

    var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterDate(period.start, period.end)
        .filterBounds(point)
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
        .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
        .select(['VV', 'VH']);

    var s1_median = s1.median().clip(qhh_border)
        .rename(['VV_' + period.name, 'VH_' + period.name]);

    var result = clipped
        .addBands(indices.NDVI)
        .addBands(indices.NDBI)
        .addBands(indices.MNDWI)
        .addBands(indices.BSI)
        .addBands(s1_median);

    return result;
}

var seasonalImages = seasonalPeriods.map(function(period) {
  return extractSeasonalFeatures(period);
});

var timeSeriesImage = ee.ImageCollection(seasonalImages).toBands().float();

var image = timeSeriesImage
    .addBands(dem_10m.rename('Elevation'))
    .addBands(demFeatures);

var classNames = Temperate_Steppe_11_0
    .merge(Alpine_Steppe_12_1)
    .merge(Alpine_Meadow_13_2)
    .merge(Alpine_Shrubland_20_3)
    .merge(Lake_31_4)
    .merge(River_32_5)
    .merge(Alpine_Desert_41_6)
    .merge(Sand_Dunes_42_7)
    .merge(River_Beach_43_8)
    .merge(Alpine_Wetland_50_9)
    .merge(Arable_Land_60_10)
    .merge(Urban_Land_70_11)
    .merge(Snow_and_Ice_80_12);

var bands = image.bandNames();

var training = image.select(bands).sampleRegions({
    collection: classNames,
    properties: ['landcover'],
    scale: 10
});

var split = 0.7;
var withRandom = training.randomColumn('random');
var trainingPartition = withRandom.filter(ee.Filter.lt('random', split));
var testingPartition = withRandom.filter(ee.Filter.gte('random', split));

var classifier = ee.Classifier.libsvm().train({
    features: trainingPartition,
    classProperty: 'landcover',
    inputProperties: bands
});

var classified = image.select(bands).classify(classifier);

var minSize = 20; 
var connectedPixels = classified.connectedPixelCount();
var smallPatchesMask = connectedPixels.lt(minSize);

var smallPatchMask = smallPatchesMask.updateMask(smallPatchesMask);

var majority = classified.reduceNeighborhood({
  reducer: ee.Reducer.mode(),
  kernel: ee.Kernel.square(04)
});

var cleaned = classified.where(smallPatchMask, majority);

var test = testingPartition.classify(classifier);
var confusionMatrix = test.errorMatrix('landcover', 'classification');
print('Confusion Matrix:', confusionMatrix);
print('Overall Accuracy:', confusionMatrix.accuracy());
print('Kappa Accuracy:', confusionMatrix.kappa());
print('Producers Accuracy:', confusionMatrix.producersAccuracy());
print('Users Accuracy:', confusionMatrix.consumersAccuracy());

Map.addLayer(cleaned, {
    min: 0, max: 12,
    palette: [
      'b3ca1f', '52f132', '015c14', '20a315', '193bd6',
      '16b7ff', 'ffdfca', 'fff518', '2bffe9', '821299',
      'ffc82d', 'ff0000', 'fcfff3']
}, 'SVM Classified (Time Series)');

print('Training sample count per class:', training.aggregate_histogram('landcover'));
