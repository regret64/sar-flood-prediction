import requests
import time
import json

BASE = "http://127.0.0.1:5000/api/predict"

cases = [
    {"lat": 16.66015, "lon": 88.37641, "name": "sea"},
    {"lat": 13.0827, "lon": 80.2707, "name": "chennai"}
]

print('Running API predict tests against', BASE)

for case in cases:
    url = f"{BASE}?lat={case['lat']}&lon={case['lon']}"
    print('\nTEST:', case['name'], case['lat'], case['lon'])
    try:
        res = requests.get(url, timeout=10)
        print('HTTP', res.status_code)
        j = res.json()
        print(json.dumps(j, indent=2))
        # basic checks
        risk = j.get('risk')
        msg = j.get('message') or j.get('msg')
        temp = j.get('temp') if 'temp' in j else j.get('temperature')
        hum = j.get('humidity') if 'humidity' in j else j.get('humidity')
        score = j.get('score')
        conf = j.get('confidence')
        print('Checks:')
        print(' - risk present:', isinstance(risk, str))
        print(' - message present:', isinstance(msg, (str, type(None))))
        print(' - temp numeric:', isinstance(temp, (int, float)) and not isinstance(temp, bool))
        print(' - humidity numeric:', isinstance(hum, (int, float)) and not isinstance(hum, bool))
        print(' - score numeric:', isinstance(score, (int, float)) and not isinstance(score, bool))
        print(' - confidence numeric:', isinstance(conf, (int, float)) and not isinstance(conf, bool))
    except Exception as e:
        print('ERROR', e)

# Determinism test: call same point multiple times
print('\nDeterminism test (Chennai)')
vals = []
for i in range(3):
    r = requests.get(f"{BASE}?lat=13.0827&lon=80.2707", timeout=10)
    try:
        j = r.json()
        vals.append(json.dumps(j, sort_keys=True))
    except Exception as e:
        vals.append(str(r.status_code))
    time.sleep(0.5)

same = all(v == vals[0] for v in vals)
print('All 3 identical:', same)
for i,v in enumerate(vals):
    print(f'--- run {i+1} ---')
    print(v)
