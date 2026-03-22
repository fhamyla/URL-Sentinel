import re
import tldextract

def extract_features(url):

    features = []

    features.append(len(url))

    features.append(url.count("."))

    features.append(1 if "@" in url else 0)

    features.append(1 if "-" in url else 0)

    features.append(1 if url.startswith("https") else 0)

    features.append(sum(c.isdigit() for c in url))

    ip_pattern = r'\d+\.\d+\.\d+\.\d+'
    features.append(1 if re.search(ip_pattern, url) else 0)

    suspicious = ["login","verify","account","update","secure","bank"]

    count = 0
    for word in suspicious:
        if word in url.lower():
            count += 1

    features.append(count)

    ext = tldextract.extract(url)
    domain = ext.domain

    features.append(len(domain))

    return features