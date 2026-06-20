# Forest Fire Prediction and Spread Simulation using AI/ML

## Overview

Forest fires pose a significant threat to biodiversity, human settlements, and environmental sustainability. Early prediction and accurate simulation of fire spread can help authorities implement preventive measures and optimize resource allocation during emergencies.

This project presents an AI-powered geospatial framework for predicting forest fire occurrences and simulating their spread using machine learning and cellular automata techniques. The system integrates weather conditions, terrain characteristics, land use information, and historical fire incidents to generate next-day fire probability maps and forecast fire propagation over multiple time horizons.

---

## Objectives

1. Predict the probability of forest fire occurrence for the next day.
2. Generate a binary classification map (Fire / No Fire).
3. Simulate fire spread for:

   * 1 Hour
   * 2 Hours
   * 3 Hours
   * 6 Hours
   * 12 Hours
4. Produce raster outputs at 30-meter spatial resolution.
5. Generate animated visualizations of fire spread progression.

---

## Problem Statement

Forest fire spread depends on multiple environmental and anthropogenic factors such as:

* Temperature
* Relative Humidity
* Rainfall
* Wind Speed and Direction
* Terrain Slope
* Terrain Aspect
* Vegetation/Fuel Availability
* Human Settlements
* Road Networks

Traditional forecasting methods struggle to model the dynamic interactions among these variables. This project leverages Artificial Intelligence and Geospatial Analytics to improve prediction accuracy and simulate realistic fire propagation scenarios.

---

## Methodology

### Phase 1: Data Collection

The following datasets are collected and preprocessed:

#### Weather Data

* ERA5 Reanalysis Data
* IMD Weather Data
* MOSDAC Products

Parameters:

* Temperature
* Rainfall
* Relative Humidity
* Wind Speed
* Wind Direction

#### Terrain Data

* DEM (Digital Elevation Model)
* Slope
* Aspect

Source:

* Bhoonidhi Portal

#### Land Cover Data

* Land Use Land Cover (LULC)
* Fuel Availability Layers

Source:

* Bhuvan
* Sentinel Hub

#### Human Influence Data

* Road Networks
* Settlement Density

Source:

* GHSL

#### Historical Fire Data

* VIIRS Active Fire Dataset

---

### Phase 2: Feature Engineering

All datasets are:

1. Reprojected to a common coordinate system.
2. Resampled to 30m spatial resolution.
3. Aligned spatially.
4. Converted into a multi-band feature stack.

Generated Features:

| Feature            | Description              |
| ------------------ | ------------------------ |
| Temperature        | Daily mean temperature   |
| Humidity           | Relative humidity        |
| Rainfall           | Daily precipitation      |
| Wind Speed         | Fire acceleration factor |
| Wind Direction     | Directional propagation  |
| Slope              | Terrain inclination      |
| Aspect             | Terrain orientation      |
| Fuel Load          | Vegetation density       |
| Settlement Density | Human-induced fire risk  |
| Distance to Roads  | Accessibility factor     |

---

### Phase 3: Fire Prediction Model

A U-Net based deep learning architecture is used for semantic segmentation of fire-prone regions.

#### Input

Multi-channel raster feature stack.

#### Output

Pixel-wise fire probability map.

#### Classes

* High Risk
* Moderate Risk
* Low Risk
* No Risk

The output probability map is converted into a binary fire/no-fire raster using a threshold.

---

### Phase 4: Fire Spread Simulation

Predicted high-risk pixels act as ignition points.

A Cellular Automata (CA) model is used to simulate fire propagation.

#### Spread Factors

##### Wind Effect

Higher wind speeds increase propagation probability.

##### Slope Effect

Fire spreads faster uphill.

##### Fuel Availability

Dense vegetation increases spread intensity.

##### Neighbor Influence

Fire can propagate to adjacent cells.

---

### Cellular Automata Rules

For each burning cell:

1. Examine neighboring cells.
2. Compute spread probability using:

   * Wind
   * Slope
   * Fuel
3. Ignite neighboring cells if probability exceeds threshold.
4. Update grid state for the next timestep.

---

## System Architecture

Raw Data Sources
↓
Weather Data
DEM Data
LULC Data
Road Networks
Historical Fire Records
↓
Data Preprocessing
↓
Feature Stack Generation
↓
U-Net Fire Prediction Model
↓
Fire Probability Raster
↓
Binary Fire Map
↓
Cellular Automata Simulation
↓
1h / 2h / 3h / 6h / 12h Forecasts
↓
Animation Generation

## Technologies Used

### Programming

* Python

### Machine Learning

* PyTorch
* TensorFlow
* Scikit-Learn

### Geospatial Processing

* GDAL
* Rasterio
* GeoPandas

### Visualization

* Matplotlib
* Folium
* ImageIO

### Deep Learning Models

* U-Net
* ConvLSTM (Future Extension)

### Simulation

* Cellular Automata

---

## Outputs

### Fire Prediction

* Fire Probability Map (.tif)
* Binary Fire Classification Map (.tif)

### Fire Spread Simulation

* spread_1h.tif
* spread_2h.tif
* spread_3h.tif
* spread_6h.tif
* spread_12h.tif

### Visualization

* Fire Spread Animation (.gif)

---

## Evaluation Metrics

### Prediction Performance

* Accuracy
* Precision
* Recall
* F1 Score
* IoU (Intersection over Union)

### Simulation Performance

* Burned Area Accuracy
* Spread Pattern Similarity
* Temporal Consistency

---

## Results

The proposed framework enables:

* Early identification of fire-prone regions.
* Next-day fire probability forecasting.
* Realistic fire spread simulation.
* Decision support for disaster management agencies.
* Enhanced forest protection and resource planning.

---

## Future Enhancements

* Attention U-Net Architecture
* ConvLSTM Temporal Prediction
* Real-time Satellite Data Integration
* Drone-based Fire Monitoring
* Explainable AI (XAI)
* WebGIS Dashboard Deployment

---

## Contributors

Developed as part of an AI/ML-based Geospatial Forest Fire Prediction and Spread Simulation Project.

## Afreed,saketh,ashraf,nikitha
## License

This project is released under the MIT License.
