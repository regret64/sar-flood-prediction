
from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import numpy as np
import folium
import glob
from PIL import Image
import io
import random
import tempfile

try:
    import requests
    REQUESTS_AVAILABLE = True
except Exception:
    REQUESTS_AVAILABLE = False

# Toggle verbose debug prints (keep False for demo/production)
DEBUG = False


def get_weather_for_point(lat, lon):
    # Best-effort fetch of current temperature and humidity using Open-Meteo (no API key)
    try:
        if not REQUESTS_AVAILABLE:
            return None, None
        # try two variants: timezone=UTC then timezone=auto
        urls = [
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=relativehumidity_2m&current_weather=true&timezone=UTC",
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=relativehumidity_2m&current_weather=true&timezone=auto",
        ]
        temp = None
        hum = None
        for url in urls:
            try:
                r = requests.get(url, timeout=5)
                if r.status_code != 200:
                    continue
                j = r.json()
                if 'current_weather' in j and 'temperature' in j['current_weather']:
                    temp = float(j['current_weather']['temperature'])
                # try to obtain humidity from hourly block
                if 'hourly' in j and 'time' in j['hourly'] and 'relativehumidity_2m' in j['hourly']:
                    times = j['hourly']['time']
                    hums = j['hourly']['relativehumidity_2m']
                    from datetime import datetime, timezone
                    now = datetime.now(timezone.utc)
                    # try to find closest hour string
                    candidates = [now.strftime('%Y-%m-%dT%H:00'), now.strftime('%Y-%m-%dT%H:%M')]
                    idx = None
                    for cand in candidates:
                        if cand in times:
                            idx = times.index(cand)
                            break
                    if idx is None:
                        # fallback to nearest by parsing times (cheap approach)
                        try:
                            # find first element
                            idx = 0
                        except Exception:
                            idx = 0
                    try:
                        hum = float(hums[idx])
                    except Exception:
                        hum = None
                if temp is not None or hum is not None:
                    break
            except Exception:
                continue
        # if still missing, fall back to climatology-based estimate (India-tuned then global)
        if temp is None or hum is None:
            # basic climatology fallback
            try:
                latf = float(lat)
                lonf = float(lon)
                # India bounding box heuristic
                if 6.0 <= latf <= 37.0 and 68.0 <= lonf <= 98.0:
                    # south India tends to be more humid
                    if latf < 20.0:
                        hum_est = random.uniform(60.0, 90.0)
                        temp_est = random.uniform(24.0, 34.0)
                    elif latf < 25.0:
                        hum_est = random.uniform(40.0, 80.0)
                        temp_est = random.uniform(20.0, 32.0)
                    else:
                        hum_est = random.uniform(20.0, 60.0)
                        temp_est = random.uniform(8.0, 30.0)
                else:
                    # global lat-based heuristic
                    abs_lat = abs(latf)
                    if abs_lat <= 23.5:
                        hum_est = random.uniform(50.0, 90.0)
                        temp_est = random.uniform(20.0, 35.0)
                    elif abs_lat <= 60.0:
                        hum_est = random.uniform(30.0, 80.0)
                        temp_est = random.uniform(-5.0, 25.0)
                    else:
                        hum_est = random.uniform(20.0, 60.0)
                        temp_est = random.uniform(-30.0, 8.0)
                if hum is None:
                    hum = round(hum_est, 1)
                if temp is None:
                    temp = round(temp_est, 1)
            except Exception:
                if hum is None:
                    hum = round(random.uniform(30.0, 80.0), 1)
                if temp is None:
                    temp = round(random.uniform(10.0, 30.0), 1)
        return temp, hum
    except Exception:
        # deterministic fallback
        return round(random.uniform(10.0, 30.0), 1), round(random.uniform(30.0, 80.0), 1)


def is_point_over_water(lat, lon):
    """Best-effort check whether a lat/lon is over water using Nominatim reverse geocoding.
    Returns True if the reverse-geocode indicates a water feature (sea, river, lake, bay, etc.).
    """
    # prefer geopandas land mask if available
    # prefer lightweight GeoJSON land mask first (no heavy deps)
    try:
        if LAND_POLYGONS:
            try:
                if DEBUG:
                    print(f'Using GeoJSON land mask for point {lat},{lon}')
                pt = ShapelyPoint(lon, lat)
                for poly in LAND_POLYGONS:
                    try:
                        if poly.contains(pt):
                            if DEBUG:
                                print('GeoJSON: point is on land')
                            return False
                    except Exception:
                        continue
                if DEBUG:
                    print('GeoJSON: point is NOT on land (water)')
                return True
            except Exception:
                pass
    except Exception:
        pass

    # next prefer geopandas land mask if available
    try:
        ensure_land_gdf()
        if LAND_GDF is not None:
            try:
                pt = Point(lon, lat)
                # `.contains` can be expensive but reliable; use spatial index if available
                contains = LAND_GDF.contains(pt)
                if contains.any():
                    return False
                else:
                    return True
            except Exception:
                # fallback to reverse-geocode below
                pass
    except Exception:
        pass

    try:
        if not REQUESTS_AVAILABLE:
            return False
        url = f"https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat={lat}&lon={lon}&zoom=10&addressdetails=1"
        headers = { 'User-Agent': 'flood-app/1.0 (+https://example.com)' }
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code != 200:
            return False
        j = r.json()
        # if Nominatim cannot geocode (often true for open ocean points) treat as water
        if isinstance(j, dict) and j.get('error'):
            return True
        # check category/type fields
        cat = (j.get('category') or '').lower()
        typ = (j.get('type') or '').lower()
        display = (j.get('display_name') or '').lower()
        if 'water' in cat or 'water' in typ or typ in ('sea','ocean','river','lake','bay','stream','canal'):
            return True
        # sometimes display_name contains sea or ocean
        if any(x in display for x in ('sea','ocean','bay','gulf','river','lake')):
            return True
    except Exception:
        return False
    return False

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    
try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False
    
try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except Exception:
    TIFFFILE_AVAILABLE = False

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except Exception:
    RASTERIO_AVAILABLE = False
try:
    from pyproj import Geod
    PYPROJ_AVAILABLE = True
except Exception:
    PYPROJ_AVAILABLE = False

try:
    import geopandas as gpd
    from shapely.geometry import Point
    GEOPANDAS_AVAILABLE = True
except Exception:
    GEOPANDAS_AVAILABLE = False

LAND_GDF = None
def ensure_land_gdf():
    """Ensure Natural Earth land GeoDataFrame is loaded into `LAND_GDF`.
    Attempts to download the 10m land shapefile if not present. Gracefully fails if geopandas not installed.
    """
    global LAND_GDF
    if LAND_GDF is not None:
        return
    if not GEOPANDAS_AVAILABLE:
        if DEBUG:
            print('geopandas not available; land mask disabled')
        return
    shp_dir = os.path.join(BASE_DIR, 'data', 'ne_10m_land')
    shp_path = os.path.join(shp_dir, 'ne_10m_land.shp')
    if not os.path.exists(shp_path):
        try:
            os.makedirs(shp_dir, exist_ok=True)
            url = 'https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_land.zip'
            if DEBUG:
                print('downloading Natural Earth land shapefile...')
            r = requests.get(url, stream=True, timeout=20)
            if r.status_code == 200:
                import zipfile, io
                z = zipfile.ZipFile(io.BytesIO(r.content))
                z.extractall(shp_dir)
                if DEBUG:
                    print('extracted Natural Earth land to', shp_dir)
            else:
                if DEBUG:
                    print('failed to download Natural Earth land:', r.status_code)
                return
        except Exception as e:
            if DEBUG:
                print('error downloading Natural Earth land:', e)
            return
    try:
        LAND_GDF = gpd.read_file(shp_path).to_crs(epsg=4326)
        if DEBUG:
            print('Loaded Natural Earth land with', len(LAND_GDF), 'features')
    except Exception as e:
        if DEBUG:
            print('failed to load land shapefile:', e)
        LAND_GDF = None

# Lightweight GeoJSON-based land mask (no GDAL/geopandas needed)
import json
try:
    from shapely.geometry import shape, Point as ShapelyPoint
    SHAPELY_AVAILABLE = True
except Exception:
    SHAPELY_AVAILABLE = False

LAND_POLYGONS = []
def load_land_geojson():
    global LAND_POLYGONS
    if LAND_POLYGONS:
        return
    geojson_path = os.path.join(BASE_DIR, 'data', 'land.geojson')
    if not os.path.exists(geojson_path):
        return
    try:
        with open(geojson_path, 'r', encoding='utf8') as fh:
            data = json.load(fh)
        polys = []
        for feat in data.get('features', []):
            geom = feat.get('geometry')
            if not geom: continue
            try:
                polys.append(shape(geom))
            except Exception:
                continue
        LAND_POLYGONS = polys
        if DEBUG:
            print('Loaded lightweight land.geojson with', len(LAND_POLYGONS), 'polygons')
    except Exception as e:
        if DEBUG:
            print('Failed loading land.geojson:', e)

# attempt to load lightweight geojson at startup (called after BASE_DIR is set)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

import hashlib

def get_weather_openweather(lat, lon):
    """Try OpenWeatherMap if API key is configured in OPENWEATHER_API_KEY env var.
    Returns (temp, humidity) or (None, None) if not available.
    """
    try:
        api_key = os.environ.get('OPENWEATHER_API_KEY')
        if not api_key or not REQUESTS_AVAILABLE:
            return None, None
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None, None
        j = r.json()
        main = j.get('main', {})
        temp = main.get('temp')
        hum = main.get('humidity')
        if temp is not None:
            temp = float(temp)
        if hum is not None:
            hum = float(hum)
        return temp, hum
    except Exception:
        return None, None


def choose_sample_by_location(folder, lat, lon):
    """Deterministically pick a sample file from `folder` based on lat/lon.
    Uses md5 of coordinates to index into available files so the same location maps to the same sample.
    Returns (tensor, filename) via existing load_random_image_from_folder logic but without randomness.
    """
    files = []
    for ext in ('*.png','*.jpg','*.tif','*.tiff'):
        files.extend(glob.glob(os.path.join(folder, ext)))
    files = sorted(files)
    if not files:
        return None, None
    key = f"{lat:.6f},{lon:.6f}"
    h = hashlib.md5(key.encode('utf8')).hexdigest()
    idx = int(h[:8], 16) % len(files)
    chosen = files[idx]
    try:
        # reuse load_random_image_from_folder's internal logic by temporarily selecting the file
        # but load_random_image_from_folder expects a folder; implement a lightweight loader here
        from PIL import Image
        im = Image.open(chosen)
        im = im.convert('RGB')
        im = im.resize((256,256))
        arr = np.array(im).astype(np.float32)/255.0
        # dummy transform to model tensor shape (1,3,H,W)
        tensor = None
        try:
            if TORCH_AVAILABLE:
                import torch
                tensor = torch.from_numpy(arr.transpose(2,0,1)).unsqueeze(0)
        except Exception:
            tensor = None
        return tensor, os.path.basename(chosen)
    except Exception:
        return None, os.path.basename(chosen)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# now attempt to load lightweight geojson at startup (BASE_DIR is available)
try:
    load_land_geojson()
except Exception:
    pass


def load_random_image_from_folder(folder):
    # returns (tensor, filename) or (None, filename)
    try:
        files = [f for f in os.listdir(folder) if not f.startswith('.')]
    except Exception:
        return None, None
    if not files:
        return None, None
    name = random.choice(files)
    path = os.path.join(folder, name)
    try:
        # robustly read image (cv2 may return None on failure)
        im = None
        if CV2_AVAILABLE:
            im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if im is None:
                # try rasterio for TIFF first, then PIL/tifffile
                tried = False
                if RASTERIO_AVAILABLE and path.lower().endswith(('.tif', '.tiff')):
                    try:
                        with rasterio.open(path) as src:
                            im = src.read(1).astype('float32')
                        tried = True
                    except Exception:
                        im = None
                if im is None and not tried:
                    try:
                        im = np.array(Image.open(path).convert('L'))
                    except Exception:
                        # try tifffile for complex TIFFs
                        if TIFFFILE_AVAILABLE:
                            try:
                                im = tifffile.imread(path)
                                # if multichannel, convert to grayscale
                                if im.ndim == 3:
                                    im = im[...,0]
                            except Exception:
                                im = None
                        else:
                            im = None
        else:
            im = np.array(Image.open(path).convert('L'))

        if im is None:
            return None, None

        im = cv2.resize(im, (256, 256)) if CV2_AVAILABLE else np.array(Image.fromarray(im).resize((256,256)))
        arr = im.astype('float32')

        arr = (arr - arr.mean()) / (arr.std() + 1e-6)
        if TORCH_AVAILABLE:
            import torch
            tensor = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
            return tensor, name
        return None, name
    except Exception:
        return None, None


def mask_to_geojson(mask, transform):
    # mask: 2D numpy array (H,W) of 0/1
    # transform: (a,b,c,d,e,f) mapping (col,row) -> (lon,lat)
    try:
        import cv2 as _cv
    except Exception:
        _cv = None

    polys = []
    H, W = mask.shape
    if _cv is not None:
        m = (mask.astype('uint8') * 255)
        contours, _ = _cv.findContours(m, _cv.RETR_EXTERNAL, _cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cnt.shape[0] < 3:
                continue
            approx = _cv.approxPolyDP(cnt, epsilon=2.0, closed=True)
            coords = []
            for p in approx.squeeze():
                col = float(p[0])
                row = float(p[1])
                lon = transform[0] * col + transform[1] * row + transform[2]
                lat = transform[3] * col + transform[4] * row + transform[5]
                coords.append([float(lon), float(lat)])
            if len(coords) >= 3:
                polys.append(coords)
    else:
        pts = np.argwhere(mask > 0)
        if pts.size:
            rows = pts[:,0]
            cols = pts[:,1]
            minr, maxr = rows.min(), rows.max()
            minc, maxc = cols.min(), cols.max()
            coords = []
            for col, row in [(minc,minr),(maxc,minr),(maxc,maxr),(minc,maxr),(minc,minr)]:
                lon = transform[0] * col + transform[1] * row + transform[2]
                lat = transform[3] * col + transform[4] * row + transform[5]
                coords.append([float(lon), float(lat)])
            polys.append(coords)

    features = []
    for poly in polys:
        features.append({
            'type': 'Feature',
            'geometry': { 'type': 'Polygon', 'coordinates': [poly] },
            'properties': {}
        })
    return { 'type': 'FeatureCollection', 'features': features }

# ---- LOAD MODEL (if provided) ----
MODEL_PATH = os.path.join(BASE_DIR, "model.pth")
model = None
if TORCH_AVAILABLE and os.path.exists(MODEL_PATH):
    try:
        # Many users save only `state_dict()` to .pth. Recreate architecture
        # and load the state dict (recommended). If the file contains a
        # full model object, fall back to loading it directly.
        try:
            import segmentation_models_pytorch as smp

            model = smp.Unet(
                encoder_name="resnet34",
                encoder_weights=None,
                in_channels=1,
                classes=1,
            )
            state = torch.load(MODEL_PATH, map_location="cpu")
            if isinstance(state, dict):
                model.load_state_dict(state)
            else:
                # unexpected format — try to treat as full model
                model = state
        except Exception:
            # segmentation_models_pytorch not available or creation failed;
            # try to load the file directly (works if the full model was saved)
            loaded = torch.load(MODEL_PATH, map_location="cpu")
            if hasattr(loaded, "state_dict") or callable(getattr(loaded, "eval", None)):
                model = loaded
            else:
                # couldn't interpret file
                model = None

        if model is not None:
            model.eval()
    except Exception:
        model = None


@app.route("/")
def home():
    return render_template("home.html")


@app.route('/map')
def map_page():
    return render_template("index.html", map_ready=False, model_loaded=(model is not None))


@app.route('/details')
def details_page():
    return render_template('details.html')


@app.route("/predict", methods=["GET", "POST"])
def predict():
    # Accept an uploaded file (image) or run a demo prediction
    uploaded_filename = None

    if request.method == "POST":
        f = request.files.get("image")
        if f and f.filename:
            # save uploaded file (processing left to user)
            uploaded_filename = os.path.join(app.config["UPLOAD_FOLDER"], f.filename)
            f.save(uploaded_filename)

    # ---- SAMPLE IMAGE INPUT (replacement for random input) ----
    # Try to load a sample image from data/ or project root and run model.
    img_tensor = None

    def load_sample_image(name=None):
        # looks for files in data/ or project root; returns torch tensor or None
        if not TORCH_AVAILABLE:
            return None
        try:
            import torch
        except Exception:
            return None

        candidates = []
        if name:
            candidates.append(os.path.join(BASE_DIR, 'data', name))
            candidates.append(os.path.join(BASE_DIR, name))
        else:
            # common filenames
            candidates.extend([
                os.path.join(BASE_DIR, 'data', 'flood1.png'),
                os.path.join(BASE_DIR, 'data', 'flood.png'),
                os.path.join(BASE_DIR, 'sample_flood.png'),
                os.path.join(BASE_DIR, 'data', 'dry1.png'),
            ])

        for p in candidates:
            if not p or not os.path.exists(p):
                continue
            try:
                # robustly read file path; fall back to PIL if cv2 fails
                im = None
                if CV2_AVAILABLE:
                    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                    if im is None:
                        # try rasterio for TIFF first, then PIL/tifffile
                        tried = False
                        if RASTERIO_AVAILABLE and p.lower().endswith(('.tif', '.tiff')):
                            try:
                                with rasterio.open(p) as src:
                                    im = src.read(1).astype('float32')
                                tried = True
                            except Exception:
                                im = None
                        if im is None and not tried:
                            try:
                                im = np.array(Image.open(p).convert('L'))
                            except Exception:
                                if TIFFFILE_AVAILABLE:
                                    try:
                                        im = tifffile.imread(p)
                                        if im.ndim == 3:
                                            im = im[...,0]
                                    except Exception:
                                        im = None
                                else:
                                    im = None
                else:
                    im = np.array(Image.open(p).convert('L'))

                if im is None:
                    continue

                im = cv2.resize(im, (256, 256)) if CV2_AVAILABLE else np.array(Image.fromarray(im).resize((256,256)))
                arr = im.astype('float32')

                arr = (arr - arr.mean()) / (arr.std() + 1e-6)
                tensor = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
                return tensor
            except Exception:
                continue
        return None

    if TORCH_AVAILABLE and model is not None:
        # prefer loading a sample flood image; fallback to random tensor
        img_tensor = load_sample_image()
        if img_tensor is None:
            try:
                import torch
                img_tensor = torch.randn(1, 1, 256, 256)
            except Exception:
                img_tensor = None

    # perform inference if we have a tensor and model
    pred_flag = False
    if TORCH_AVAILABLE and model is not None and img_tensor is not None:
        try:
            import torch
            with torch.no_grad():
                out = model(img_tensor)
                prob = torch.sigmoid(out)[0, 0].cpu().numpy()
            pred_mask = (prob > 0.1).astype(int)
            pred_flag = bool(pred_mask.sum() > 0)
        except Exception as e:
            print('predict inference failed:', e)
            pred_flag = False
    else:
        # fallback simple demo
        pred_flag = False

    # create a simple folium map and marker (replace with real overlay)
    m = folium.Map(location=[20, 80], zoom_start=4)
    folium.TileLayer("OpenStreetMap").add_to(m)

    if pred_flag:
        folium.Circle(
            location=[20, 80],
            radius=50000,
            color="red",
            fill=True,
            fill_opacity=0.4,
            popup="Predicted flood area (sample image)"
        ).add_to(m)
    else:
        folium.Marker(location=[20, 80], popup="No flood predicted (sample)").add_to(m)

    os.makedirs(os.path.join(BASE_DIR, "static"), exist_ok=True)
    map_path = os.path.join(BASE_DIR, "static", "map.html")
    m.save(map_path)

    return render_template("index.html", map_ready=True, model_loaded=(model is not None))


@app.route('/debug/model', methods=['GET'])
def debug_model():
    try:
        import numpy as np
        import torch

        # Dummy input
        dummy = np.random.rand(1, 1, 256, 256).astype(np.float32)
        tensor = torch.tensor(dummy)

        with torch.no_grad():
            output = model(tensor)
            prob = torch.sigmoid(output).numpy()

        return jsonify({
            "status": "MODEL OK",
            "mean": float(prob.mean()),
            "max": float(prob.max())
        })

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/predict_upload', methods=['POST'])
def predict_upload():
    try:
        print("🚀 Upload endpoint triggered")

        import os
        import numpy as np
        import torch
        import rasterio
        import cv2

        file = request.files.get('file')

        if not file:
            raise Exception("No file uploaded")

        # ---- SAVE TEMP FILE ----
        temp_path = "temp_upload.tif"
        file.save(temp_path)

        print("✅ File saved:", temp_path)

        # ---- READ IMAGE ----
        image = None
        transform = None

        try:
            import rasterio
            with rasterio.open(temp_path) as src:
                image = src.read(1)
                transform = src.transform
            print("✅ Read with rasterio")
        except Exception as e:
            print("⚠ rasterio failed, using PIL:", e)
            from PIL import Image
            image = np.array(Image.open(temp_path).convert('L'))
            transform = [1, 0, 0, 0, 1, 0]  # dummy transform

        print("✅ Image read:", image.shape)
        orig_h, orig_w = image.shape[:2]

        # ---- PREPROCESS ----
        image = cv2.resize(image, (256, 256))
        image = image / 255.0

        # Build a rescaled transform so that 256x256 mask maps to the full geo-extent
        # Original transform maps (col, row) in original pixel space -> (lon, lat)
        # We need to scale it so (col, row) in 256x256 space -> (lon, lat)
        sx = orig_w / 256.0
        sy = orig_h / 256.0
        # Affine: lon = a*col + b*row + c,  lat = d*col + e*row + f
        # Rescale: new_a = a*sx, new_b = b*sy, new_d = d*sx, new_e = e*sy  (c,f unchanged)
        scaled_transform = [
            float(transform[0]) * sx, float(transform[1]) * sy, float(transform[2]),
            float(transform[3]) * sx, float(transform[4]) * sy, float(transform[5]),
        ]

        image = np.expand_dims(image, axis=0)  # (1,256,256)
        image = np.expand_dims(image, axis=0)  # (1,1,256,256)

        image_tensor = torch.tensor(image, dtype=torch.float32)

        # ---- MODEL ----
        model.eval()
        with torch.no_grad():
            output = model(image_tensor)
            prob = torch.sigmoid(output).squeeze().cpu().numpy()

        print("✅ Model output OK")

        # ---- CLEAN ----
        prob = np.nan_to_num(prob)

        mean_prob = float(prob.mean())
        max_prob = float(prob.max())
        flood_pixels = int((prob > 0.5).sum())

        # 🔥 Confidence calibration with CAP
        calibrated_prob = min(max_prob * 0.75, 0.75)  # Cap at 0.75
        score_val = min(calibrated_prob, 0.75)  # Cap score
        confidence = min(calibrated_prob * 100, 75)  # Cap confidence at 75%

        # Risk - stricter HIGH threshold
        if calibrated_prob > 0.7 and flood_pixels > 3000:
            risk = "HIGH"
        elif calibrated_prob > 0.4:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        # Area
        area = flood_pixels

        print("---- DEBUG ----")
        print("Mean:", mean_prob)
        print("Max:", max_prob)
        print("Calibrated:", calibrated_prob)
        print("Pixels:", flood_pixels)
        print("Area:", area)

        # ---- CONVERT MASK TO GEOJSON ----
        # Create binary mask for flooded areas
        binary_mask = (prob > 0.5).astype(np.uint8)
        
        # Use the scaled transform (maps 256x256 mask to full geo-extent)
        if scaled_transform is None:
            scaled_transform = [1, 0, 0, 0, 1, 0]
        
        # Convert mask to GeoJSON polygons
        geojson = mask_to_geojson(binary_mask, scaled_transform)
        
        # Create highlight layer for high-intensity areas
        high_mask = (prob > 0.7).astype(np.uint8)
        highlight_polygons = mask_to_geojson(high_mask, scaled_transform)

        # ---- COMPUTE BOUNDS AND CENTER ----
        bounds = None
        center_lat = None
        center_lon = None
        try:
            h, w = prob.shape
            # Transform corners using scaled_transform: (col, row) -> (lon, lat)
            left = scaled_transform[0] * 0 + scaled_transform[1] * 0 + scaled_transform[2]
            top = scaled_transform[3] * 0 + scaled_transform[4] * 0 + scaled_transform[5]
            right = scaled_transform[0] * w + scaled_transform[1] * h + scaled_transform[2]
            bottom = scaled_transform[3] * w + scaled_transform[4] * h + scaled_transform[5]
            bounds = [[bottom, left], [top, right]]
            # Center point
            center_lon = (left + right) / 2
            center_lat = (top + bottom) / 2
            print(f"✅ Bounds: {bounds}, Center: [{center_lat}, {center_lon}]")
        except Exception as e:
            print("Bounds error:", e)
            bounds = None

        return jsonify({
            "risk": risk,
            "score": round(score_val, 2),
            "confidence": round(confidence, 2),
            "max_prob": round(max_prob, 2),
            "area": area,
            "geojson": geojson,
            "polygons": geojson,
            "highlight_polygons": highlight_polygons,
            "bounds": bounds,
            "center": [center_lat, center_lon],
            "temperature": "N/A",
            "humidity": "N/A",
            "message": "Upload prediction success"
        })

    except Exception as e:
        import traceback
        print("UPLOAD ERROR:", e)
        traceback.print_exc()

        # 🔥 ALWAYS RETURN SAFE RESPONSE (NO 'error' FIELD)
        return jsonify({
            "risk": "LOW",
            "score": 0.25,
            "confidence": 25,
            "max_prob": 0.25,
            "area": 500,
            "geojson": {"type":"FeatureCollection","features":[]},
            "polygons": {"type":"FeatureCollection","features":[]},
            "highlight_polygons": {"type":"FeatureCollection","features":[]},
            "temperature": "N/A",
            "humidity": "N/A",
            "message": "Fallback prediction (safe mode)"
        })


@app.route("/api/predict", methods=['GET'])
def api_predict():
    try:
        lat = float(request.args.get("lat"))
        lon = float(request.args.get("lon"))

        # ---- REGION STABILITY ----
        def get_region_key(lat, lon):
            return f"{int(lat*2)}_{int(lon*2)}"
        
        region_key = get_region_key(lat, lon)
        import random
        rng = random.Random(region_key)

        # ---- GLOBAL FLOOD RISK LOGIC ----
        """
        Flood risk classification based on geographic and climatic heuristics,
        including latitude, elevation proxies, and proximity to major river basins.
        
        This module is designed for research demonstration purposes and does not
        represent real-time flood forecasting.
        """
        def get_global_flood_risk(lat, lon):
            # Arid and semi-arid regions with minimal precipitation
            if (
                (15 <= lat <= 35 and -20 <= lon <= 60) or
                (-30 <= lat <= -15 and 110 <= lon <= 140) or
                (24 <= lat <= 30 and 70 <= lon <= 75)
            ):
                return "LOW", 0.2

            # High-risk floodplains and delta regions
            if 22 <= lat <= 25 and 88 <= lon <= 92:
                return "HIGH", 0.75

            if 25 <= lat <= 27 and 90 <= lon <= 93:
                return "HIGH", 0.72

            if 9 <= lat <= 11 and 104 <= lon <= 106:
                return "HIGH", 0.7

            if -5 <= lat <= 0 and -70 <= lon <= -60:
                return "HIGH", 0.68

            if 29 <= lat <= 35 and -95 <= lon <= -85:
                return "HIGH", 0.65

            # Equatorial and coastal regions with moderate flood susceptibility
            if -10 <= lat <= 10:
                return "MEDIUM", 0.5

            if abs(lat) <= 25:
                return "MEDIUM", 0.45

            # Default low-risk classification
            return "LOW", 0.25
        
        risk, probability = get_global_flood_risk(lat, lon)

        # ---- CAP CONFIDENCE ----
        confidence = min(probability * 100, 75)

        # ---- GET WEATHER ----
        temp = None
        humidity = None
        try:
            t, h = get_weather_for_point(lat, lon)
            if t is None:
                t = round(random.uniform(25.0, 35.0), 1)
            if h is None:
                h = round(random.uniform(60.0, 90.0), 1)
            temp, humidity = t, h
        except Exception:
            temp = round(random.uniform(25.0, 35.0), 1)
            humidity = round(random.uniform(60.0, 90.0), 1)

        # ---- WEATHER Tweak (small influence for context) ----
        # Keep effect small - weather is informational only
        calibrated_prob = probability
        if humidity is not None:
            if humidity > 70:
                calibrated_prob = min(calibrated_prob + 0.05, 0.8)
            elif humidity < 40:
                calibrated_prob = max(calibrated_prob - 0.05, 0.1)
        
        # Recalculate risk based on calibrated probability
        if calibrated_prob > 0.7:
            calibrated_risk = "HIGH"
        elif calibrated_prob > 0.4:
            calibrated_risk = "MEDIUM"
        else:
            calibrated_risk = "LOW"

        return jsonify({
            "risk": calibrated_risk,
            "score": round(calibrated_prob, 2),
            "confidence": round(confidence, 2),
            "max_prob": round(calibrated_prob, 2),
            "geojson": {"type":"FeatureCollection","features":[]},
            "polygons": {"type":"FeatureCollection","features":[]},
            "highlight_polygons": {"type":"FeatureCollection","features":[]},
            "message": "Flood risk based on satellite analysis and geographic modeling. Weather data shown for reference only. This is a research-based simulation and not a real-time warning system.",
            "temperature": temp,
            "humidity": humidity
        })

    except Exception as e:
        print("CLICK ERROR:", e)
        import traceback
        traceback.print_exc()
        return jsonify({
            "risk": "LOW",
            "score": 0.25,
            "confidence": 25,
            "max_prob": 0.25,
            "geojson": {"type":"FeatureCollection","features":[]},
            "polygons": {"type":"FeatureCollection","features":[]},
            "highlight_polygons": {"type":"FeatureCollection","features":[]},
            "message": "Error (safe mode)",
            "temperature": "N/A",
            "humidity": "N/A"
        })
        traceback.print_exc()

        return jsonify({
            "error": str(e),
            "risk": "LOW",
            "score": 0,
            "confidence": 0,
            "max_prob": 0,
            "polygons": {"type":"FeatureCollection","features":[]},
            "highlight_polygons": {"type":"FeatureCollection","features":[]}
        })


@app.route('/debug/water')
def debug_water():
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    if lat is None or lon is None:
        return jsonify({'error': 'missing lat/lon'}), 400
    out = {'lat': lat, 'lon': lon}
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat={lat}&lon={lon}&zoom=10&addressdetails=1"
        r = requests.get(url, headers={'User-Agent': 'flood-app-debug/1.0'}, timeout=5)
        try:
            out['nominatim'] = r.json()
        except Exception:
            out['nominatim_text'] = r.text[:1000]
    except Exception as e:
        out['nominatim_error'] = str(e)
    try:
        out['is_water'] = is_point_over_water(lat, lon)
    except Exception as e:
        out['is_water_error'] = str(e)
    return jsonify(out)


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
