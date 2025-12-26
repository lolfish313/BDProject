from serpapi import GoogleSearch

params = {
  "engine": "google_light",
  "q": "AK104 宫颈癌 中国NMPA治疗获批 一线/二线/三线？",
  "api_key": "f526cdf7f7b124d44698c73e899f5cfe9454faecfeb3fc7612eca7fe2ea3e397"
}

search = GoogleSearch(params)
results = search.get_dict()
print(results['organic_results'])