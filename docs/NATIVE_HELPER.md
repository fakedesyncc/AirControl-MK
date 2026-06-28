# AirControl Native Helper

`aircontrol-helper` - маленький Go-бинарь без внешних зависимостей. Он не
заменяет Python-приложение и не управляет компьютером напрямую. Его задача -
быстро собрать низкоуровневую диагностику ОС, когда пользователю сложно
объяснять проблему вручную.

## Зачем Он Нужен

- Проверить графическую сессию Linux: X11, Wayland или headless.
- Найти `/dev/video*` устройства на Linux.
- Проверить наличие системных утилит: `xdotool`, `ydotool`, `v4l2-ctl`, `flac`.
- Дать рекомендации для support ZIP и `doctor`.
- Работать как отдельный бинарь внутри PyInstaller-бандла.

Для обычного пользователя это остаётся невидимой частью приложения. Если helper
есть в бандле, `python -m aircontrol doctor` добавит блок `Native helper: OK`, а
support ZIP получит файл `native-helper.json`.

## Локальная Сборка

```bash
go test ./cmd/aircontrol-helper
mkdir -p bin
go build -trimpath -ldflags="-s -w" -o bin/aircontrol-helper ./cmd/aircontrol-helper
./bin/aircontrol-helper doctor --json
```

На Windows имя бинаря должно быть `bin/aircontrol-helper.exe`.

## Как Он Попадает В Релиз

GitHub Actions собирает helper на каждой ОС перед PyInstaller. `aircontrol.spec`
добавляет `bin/aircontrol-helper` или `bin/aircontrol-helper.exe` в frozen-бандл.
`tools/smoke_build.py` и `tools/verify_release_artifacts.py` проверяют, что
helper не потерялся в архивах и пакетах.

Если Go не установлен у разработчика, запуск из исходников остаётся рабочим:
`doctor` покажет `Native helper: missing (optional)` и продолжит Python-проверки.
