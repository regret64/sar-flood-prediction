import requests

print('Querying debug endpoint for sea coord')
print(requests.get('http://127.0.0.1:5000/debug/water?lat=16.66015&lon=88.37641', timeout=10).text)
print('\nQuerying debug endpoint for Chennai')
print(requests.get('http://127.0.0.1:5000/debug/water?lat=13.0827&lon=80.2707', timeout=10).text)
