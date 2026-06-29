# Security Policy

AirControl controls the mouse and keyboard, so input safety issues are treated
as security-adjacent even when they are not traditional vulnerabilities.

## Supported Versions

Only the latest tagged release is actively supported.

## Reporting a Problem

Please open a GitHub issue for regular bugs. For sensitive reports, avoid
posting private logs, personal datasets, medical details or camera recordings.

Useful details:

- operating system and display session, for example Windows, macOS, Linux Xorg or Linux Wayland;
- AirControl version or downloaded artifact;
- whether the app was in `View`, `Safe` or `Control`;
- `doctor-summary.txt` from a support bundle, with personal paths removed if needed;
- whether the problem is camera access, gesture detection, cursor movement, clicks or keyboard input.

## Sensitive Areas

Please report carefully if you find any of these:

- real input sent while `Safe` or `View` mode is active;
- stuck mouse button, modifier key or repeated dwell-click after losing tracking;
- command execution or file access beyond the documented diagnostics;
- support bundles containing secrets, tokens or private user data;
- Linux input backend behavior that bypasses expected permission boundaries.

## Data Handling

Do not attach raw camera recordings, personal gesture datasets or medical
information unless it is strictly necessary and intentionally shared. The project
does not need medical details to fix most accessibility and input issues.

