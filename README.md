# 🌊 Flood Prediction System using SAR Imagery

A research-based flood detection and risk assessment system using Synthetic Aperture Radar (SAR) imagery and deep learning.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Flask](https://img.shields.io/badge/Flask-2.0+-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)

> ⚠️ **Disclaimer**: This system is for **research and demonstration purposes only**. It does not represent real-time flood forecasting.

## 👥 Authors

- **Author 1** - [GitHub Profile]
- **Author 2** - [GitHub Profile]  
- **Author 3** - [GitHub Profile]

---

## Flood Prediction Demo (flask + folium)

This small app demonstrates a flood prediction demo using Flask and Folium. It runs a demo prediction by default; replace the dummy prediction with your real model and preprocessing.

Quick start
1. Create a virtual environment and install requirements:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. Place your PyTorch model as `model.pth` in the `flood_app/` folder (optional).

3. Run the app:

```bash
python app.py
```

4. Open http://127.0.0.1:5000

Notes for integrating your model
- Implement proper preprocessing for SAR or other imagery.
- Replace the demo `out = np.array([[0.2]])` block in `app.py` with real inference code using `model`.
- Save any overlays to a Folium map (geojson/polygons) and write to `static/map.html`.

Model (state_dict) loading example

If your `model.pth` is a `state_dict` (common when saving from Colab), recreate the model and load the state dict:

```python
import segmentation_models_pytorch as smp
import torch

model = smp.Unet(
	encoder_name="resnet34",
	encoder_weights=None,
	in_channels=1,
	classes=1,
)

state = torch.load("model.pth", map_location="cpu")
model.load_state_dict(state)
model.eval()
```

If the file contains a full model object (less common), `torch.load("model.pth")` may return the model directly. The app attempts the `state_dict` path first and falls back to direct loading.

Deployment
- Push the folder to GitHub and deploy to Render or Railway.
- On Render, use a Python web service with `python app.py` as the start command.
 
Optional GIS Enhancement
- This project supports a lightweight GeoJSON-based land mask to improve water/land detection without heavy GIS dependencies.
- Place a `land.geojson` file at `data/land.geojson` (a simplified Natural Earth conversion works well).
- The app will load `data/land.geojson` automatically on startup if present and use it to decide whether a clicked point is land or water.
- For higher-precision workflows you can enable the `geopandas` path (requires GDAL/Fiona) — use Conda to install `geopandas` reliably.

Fallback behavior
- If `data/land.geojson` is not present and `geopandas` is not installed, the app falls back to reverse-geocoding (Nominatim) for water detection.
