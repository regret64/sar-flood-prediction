import requests, json
lat=16.66015; lon=88.37641
for zoom in [3,4,5,6,8,10,12]:
    url=f"https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat={lat}&lon={lon}&zoom={zoom}&addressdetails=1"
    print('\nZOOM',zoom,'URL',url)
    try:
        r=requests.get(url, headers={'User-Agent':'flood-app-check/1.0'}, timeout=10)
        print('HTTP', r.status_code)
        try:
            j=r.json()
            print(json.dumps(j, indent=2))
        except Exception as e:
            print('json error', e, r.text[:500])
    except Exception as e:
        print('ERR', e)
