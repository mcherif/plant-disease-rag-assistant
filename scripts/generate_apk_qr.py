"""
Generate a QR code PNG that links to a hosted APK URL.
Usage:
    python scripts/generate_apk_qr.py --url https://example.com/plant-disease-mobile-release.apk --out apk_download_qr.png
Requires:
    pip install qrcode[pil]
"""

import argparse

import qrcode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Public URL to the APK")
    parser.add_argument("--out", default="apk_download_qr.png", help="Output PNG path")
    args = parser.parse_args()

    img = qrcode.make(args.url)
    img.save(args.out)
    print(f"Saved QR: {args.out}")
    print(f"URL: {args.url}")


if __name__ == "__main__":
    main()
