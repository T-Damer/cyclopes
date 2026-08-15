#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

OUT_FILE="cyclopes-mozilla-source.zip"
TMP_LIST="$(mktemp)"
trap 'rm -f "$TMP_LIST"' EXIT

{
  git ls-files
  printf '%s\n' \
    extension/background-firefox.html \
    extension/icons/logoOldRound.png \
    extension/manifest-firefox.json \
    extension/src/background-firefox.js \
    extension/store/promo/edge-promotional-tile-440x280.png \
    extension/store/screenshots/1.png \
    extension/store/screenshots/2.png \
    extension/store/texts/desccription.md \
    extension/store/texts/justifications.md \
    extension/store/texts/mozilla.md \
    extension/store/texts/test-instructions.md \
    scripts/build-firefox.mjs
} | sort -u > "$TMP_LIST"

zip -r -q "$OUT_FILE" -@ < "$TMP_LIST"

ZIP_SIZE=$(wc -c < "$OUT_FILE")
SHA256=$(shasum -a 256 "$OUT_FILE" | awk '{print $1}')

printf '%s\n' "Created $OUT_FILE"
printf 'Size: %s bytes\n' "$ZIP_SIZE"
printf 'SHA256: %s\n' "$SHA256"
printf 'Entries: '
zipinfo -1 "$OUT_FILE" | wc -l | tr -d '\n'
printf '\n'
