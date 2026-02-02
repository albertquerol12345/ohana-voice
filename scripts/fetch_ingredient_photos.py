import json
import os
import time
import urllib.parse
import urllib.request


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "frontend", "assets", "ingredients")
ATTR_PATH = os.path.join(OUT_DIR, "ATTRIBUTION.txt")

ITEMS = [
    ("bread.jpg", "burger bun close up"),
    ("beef.jpg", "beef burger patty close up"),
    ("chicken.jpg", "breaded chicken fillet"),
    ("vegan.jpg", "vegan burger patty"),
    ("cheese.jpg", "cheddar cheese slice"),
    ("onion.jpg", "sliced onion rings"),
    ("lettuce.jpg", "lettuce leaves"),
    ("tomato.jpg", "sliced tomato"),
    ("pickle.jpg", "pickles sliced"),
    ("bacon.jpg", "crispy bacon strips"),
    ("egg.jpg", "fried egg"),
    ("sauce.jpg", "sauce on spoon"),
    ("mushroom.jpg", "sliced mushrooms"),
    ("pepper.jpg", "jalapeno slices"),
    ("avocado.jpg", "sliced avocado"),
    ("carrot.jpg", "shredded carrot"),
    ("sweet.jpg", "honey drizzle"),
    ("generic.jpg", "burger ingredients"),
]

UNSPLASH_FALLBACK = {
    "avocado.jpg": "https://source.unsplash.com/400x400/?avocado",
    "carrot.jpg": "https://source.unsplash.com/400x400/?carrot",
    "generic.jpg": "https://source.unsplash.com/400x400/?burger,ingredients",
}


def fetch_thumb_url(query: str, width: int = 256) -> dict:
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": "6",
        "gsrlimit": "1",
        "prop": "imageinfo",
        "iiprop": "url",
        "iiurlwidth": str(width),
        "origin": "*",
        "format": "json",
    }
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "OhanaVoiceMVP/1.0 (local test)"},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    pages = payload.get("query", {}).get("pages", {})
    for page in pages.values():
        info = page.get("imageinfo", [])
        if not info:
            continue
        entry = info[0]
        return {
            "title": page.get("title"),
            "pageid": page.get("pageid"),
            "thumburl": entry.get("thumburl"),
            "url": entry.get("url"),
        }
    return {}


def download(url: str, dest: str):
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "OhanaVoiceMVP/1.0 (local test)"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp, open(dest, "wb") as f:
        f.write(resp.read())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    attributions = []
    for filename, query in ITEMS:
        dest = os.path.join(OUT_DIR, filename)
        if os.path.exists(dest):
            continue
        info = fetch_thumb_url(query)
        thumb = info.get("thumburl")
        if not thumb:
            print(f"Skip {filename} (no result for '{query}')")
            continue
        saved = False
        for attempt in range(3):
            try:
                download(thumb, dest)
                attributions.append(
                    {
                        "file": filename,
                        "query": query,
                        "title": info.get("title"),
                        "pageid": info.get("pageid"),
                        "url": info.get("url"),
                    }
                )
                print(f"Saved {filename}")
                saved = True
                break
            except Exception as exc:
                if attempt == 2:
                    print(f"Failed {filename}: {exc}")
                time.sleep(1.5)
        time.sleep(1.2)
        if not saved and filename in UNSPLASH_FALLBACK:
            try:
                download(UNSPLASH_FALLBACK[filename], dest)
                print(f"Saved {filename} (unsplash fallback)")
            except Exception as exc:
                print(f"Failed {filename} fallback: {exc}")

    if attributions:
        with open(ATTR_PATH, "w", encoding="utf-8") as f:
            for item in attributions:
                f.write(
                    f"{item['file']} | {item['title']} | {item['url']} | query={item['query']}\n"
                )


if __name__ == "__main__":
    main()
