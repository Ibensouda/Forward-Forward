import requests

url = "https://aes.cryptohack.org/ecbcbcwtf/decrypt/"
url2 = "https://aes.cryptohack.org/ecbcbcwtf/encrypt_flag/"
r = requests.get(url2)
print(r.text)
print(r.text[15+32:15+64])
temp = r.text[-35:-3]
temp2 = r.text[-35-32:-3-32]
print(requests.get(url + temp).text)
print(r.text[15:15+32])
print(requests.get(url + temp2).text)