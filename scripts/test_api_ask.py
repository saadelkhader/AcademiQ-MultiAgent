import requests, time, sys

def test():
    url = 'http://127.0.0.1:8000/ask'
    payload = {
        'query': "Test connectivity: what's maintenance predictive?",
        'session_id': 'test-ui',
        'top_k_docs': 3
    }
    start = time.time()
    try:
        r = requests.post(url, json=payload, timeout=10)
        elapsed = time.time() - start
        print('status_code:', r.status_code)
        print('elapsed_s:', round(elapsed,3))
        print('text:', r.text[:1000])
    except Exception as e:
        print('error:', str(e))

if __name__ == '__main__':
    test()
