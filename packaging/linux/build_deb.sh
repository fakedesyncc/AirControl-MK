#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIST_DIR="$ROOT/dist/AirControl"
VERSION="${AIRCONTROL_VERSION:-1.0.0}"
ARCH="${AIRCONTROL_DEB_ARCH:-amd64}"
PKG_ROOT="$ROOT/build/deb/aircontrol_${VERSION}_${ARCH}"
OUTPUT="$ROOT/AirControl-Linux-${ARCH}.deb"

if [[ ! -x "$DIST_DIR/AirControl" ]]; then
  echo "Missing PyInstaller output: $DIST_DIR/AirControl" >&2
  exit 1
fi

if ! command -v dpkg-deb >/dev/null 2>&1; then
  echo "dpkg-deb is required to build the Debian package" >&2
  exit 1
fi

rm -rf "$PKG_ROOT" "$OUTPUT"
mkdir -p "$PKG_ROOT/DEBIAN" \
         "$PKG_ROOT/opt/aircontrol" \
         "$PKG_ROOT/usr/bin" \
         "$PKG_ROOT/usr/share/applications" \
         "$PKG_ROOT/usr/share/icons/hicolor/scalable/apps"

cp -a "$DIST_DIR" "$PKG_ROOT/opt/aircontrol/AirControl"
cp "$ROOT/packaging/linux/aircontrol.svg" \
   "$PKG_ROOT/usr/share/icons/hicolor/scalable/apps/aircontrol.svg"
ln -s /opt/aircontrol/AirControl/AirControl "$PKG_ROOT/usr/bin/aircontrol"

cat > "$PKG_ROOT/usr/share/applications/aircontrol.desktop" <<'DESKTOP'
[Desktop Entry]
Type=Application
Name=AirControl
Comment=Бесконтактное ассистивное управление компьютером
Exec=/opt/aircontrol/AirControl/AirControl
Icon=aircontrol
Terminal=false
Categories=Accessibility;Utility;
Keywords=accessibility;assistive;camera;gesture;hands;
StartupWMClass=AirControl
DESKTOP

cat > "$PKG_ROOT/DEBIAN/control" <<CONTROL
Package: aircontrol
Version: ${VERSION}
Section: utils
Priority: optional
Architecture: ${ARCH}
Maintainer: AirControl <support@example.invalid>
Depends: libgl1, libgles2, libegl1, libglib2.0-0, xdotool
Description: Assistive hands-free computer control
 AirControl lets people control the computer with a webcam, hand gestures,
 dwell-click and optional voice commands. The package includes the Python
 runtime and application dependencies needed by the end-user program.
CONTROL

cat > "$PKG_ROOT/DEBIAN/postinst" <<'POSTINST'
#!/bin/sh
set -e
if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database /usr/share/applications || true
fi
if command -v gtk-update-icon-cache >/dev/null 2>&1; then
  gtk-update-icon-cache -q /usr/share/icons/hicolor || true
fi
exit 0
POSTINST

chmod 0755 "$PKG_ROOT/DEBIAN/postinst"
chmod 0644 "$PKG_ROOT/DEBIAN/control" \
           "$PKG_ROOT/usr/share/applications/aircontrol.desktop" \
           "$PKG_ROOT/usr/share/icons/hicolor/scalable/apps/aircontrol.svg"
chmod 0755 "$PKG_ROOT/opt/aircontrol/AirControl/AirControl"

dpkg-deb --build --root-owner-group "$PKG_ROOT" "$OUTPUT"
echo "Created $OUTPUT"
