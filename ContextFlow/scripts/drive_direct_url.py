"""Resolve a Google Drive file id to a direct download URL + cookies.

Prints two lines: the final URL and the Cookie header value. Use with aria2c:
  aria2c -x8 -s8 --header="Cookie: $COOKIE" "$URL" -o out
"""

import re
import sys

import requests

URL = "https://drive.usercontent.google.com/download"


def resolve(file_id: str):
    s = requests.Session()
    resp = s.get(URL, params={"id": file_id, "export": "download", "confirm": "t"}, stream=True)
    resp.raise_for_status()

    # If we got HTML, the file needs the confirm-token flow (large files).
    content_type = resp.headers.get("Content-Type", "")
    if "text/html" in content_type:
        html = resp.text
        m = re.search(r'name="confirm" value="([^"]+)"', html) or re.search(
            r'confirm=([0-9A-Za-z_-]+)', html
        )
        if m:
            token = m.group(1)
            resp = s.get(
                URL,
                params={"id": file_id, "export": "download", "confirm": token},
                stream=True,
            )
        else:
            # New-style confirmation: look for the download-form URL.
            m = re.search(r'href="(/uc\?export=download[^"]+)', html)
            if m:
                resp = s.get("https://drive.google.com" + m.group(1).replace("&amp;", "&"), stream=True)
            else:
                uuid_m = re.search(r'name="uuid" value="([^"]+)"', html)
                if uuid_m:
                    resp = s.post(
                        URL,
                        params={"id": file_id, "export": "download", "confirm": "t", "uuid": uuid_m.group(1)},
                        stream=True,
                    )
                else:
                    print("ERROR: could not find confirm token", file=sys.stderr)
                    sys.exit(1)

    cookies = "; ".join(f"{c.name}={c.value}" for c in s.cookies)
    print(resp.url)
    print(cookies)


if __name__ == "__main__":
    resolve(sys.argv[1])
