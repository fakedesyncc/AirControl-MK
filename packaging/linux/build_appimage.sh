#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APPDIR="$ROOT/AppDir"
DIST_DIR="$ROOT/dist/AirControl"
APPIMAGE="$ROOT/AirControl-Linux-x86_64.AppImage"
APPIMAGETOOL="$ROOT/appimagetool-x86_64.AppImage"

if [[ ! -x "$DIST_DIR/AirControl" ]]; then
  echo "Missing PyInstaller output: $DIST_DIR/AirControl" >&2
  exit 1
fi

rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin" \
         "$APPDIR/usr/share/applications" \
         "$APPDIR/usr/share/icons/hicolor/scalable/apps"

cp -a "$DIST_DIR" "$APPDIR/usr/bin/AirControl"
cp "$ROOT/packaging/linux/AppRun" "$APPDIR/AppRun"
cp "$ROOT/packaging/linux/AirControl.desktop" "$APPDIR/AirControl.desktop"
cp "$ROOT/packaging/linux/AirControl.desktop" "$APPDIR/usr/share/applications/AirControl.desktop"
cp "$ROOT/packaging/linux/aircontrol.svg" "$APPDIR/aircontrol.svg"
cp "$ROOT/packaging/linux/aircontrol.svg" "$APPDIR/usr/share/icons/hicolor/scalable/apps/aircontrol.svg"

chmod +x "$APPDIR/AppRun" "$APPDIR/usr/bin/AirControl/AirControl"

if [[ ! -x "$APPIMAGETOOL" ]]; then
  curl -L "https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage" \
    -o "$APPIMAGETOOL"
  chmod +x "$APPIMAGETOOL"
fi

rm -f "$APPIMAGE"
ARCH=x86_64 APPIMAGE_EXTRACT_AND_RUN=1 "$APPIMAGETOOL" "$APPDIR" "$APPIMAGE"
echo "Created $APPIMAGE"
