import ollama, time
print('ollama module:', ollama)
print('available functions:', [k for k in dir(ollama) if not k.startswith('_')])
try:
    print('Calling ollama.chat...')
    resp = ollama.chat(model='llama3', messages=[{'role':'system','content':'You are a test.'},{'role':'user','content':'Hello'}], timeout=10)
    print('resp keys:', resp.keys())
    print('message:', resp.get('message'))
except Exception as e:
    print('error:', e)
