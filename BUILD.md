# Сборка AirControl

AirControl распространяется как standalone-приложение. Пользователю не нужен
Python, компилятор или консоль: PyInstaller кладёт runtime, зависимости и модель
руки внутрь сборки.

Релизная сборка дополнительно включает маленький Go helper
`aircontrol-helper`: он собирает native diagnostics по ОС, Linux-сессии,
камерам и системным утилитам. Для пользователя это часть готового бандла.

## Что Получается

| Платформа     | Артефакт                           | Назначение                        |
| ------------- | ---------------------------------- | --------------------------------- |
| Windows       | `AirControl-Setup.exe`             | обычный установщик                |
| Windows       | `AirControl-Windows.zip`           | portable-проверка                 |
| macOS         | `AirControl-macOS.zip`             | `.app`-сборка                     |
| Debian/Ubuntu | `AirControl-Linux-amd64.deb`       | установка через пакетный менеджер |
| Linux         | `AirControl-Linux-x86_64.AppImage` | portable-запуск                   |
| Linux         | `AirControl-Linux.tar.gz`          | ручная диагностика                |

## GitHub Actions

Workflows:

- `.github/workflows/ci.yml` - быстрые проверки исходников на Ubuntu;
- `.github/workflows/build.yml` - тяжёлая сборка установщиков и архивов под
  Windows, macOS и Linux.

Оба workflow запускаются:

- при push в `main`;
- при pull request в `main`;
- вручную через `Actions -> Build AirControl -> Run workflow`.

Дополнительно `Build AirControl` запускается при push тега `v*`, например
`v2.2.0`.

Важно: GitHub показывает и запускает workflow как обычный проектный workflow
только после того, как файл `.github/workflows/build.yml` находится в default
branch репозитория. Если workflow лежит только во временной ветке, вкладка
Actions может ничего не собрать.

## Локальная Проверка Исходников

```bash
python -m pip install --upgrade pip
pip install -e .
python tools/check_tracked_sources.py
go test ./cmd/aircontrol-helper
go build -trimpath -ldflags="-s -w" -o bin/aircontrol-helper ./cmd/aircontrol-helper
python -m compileall aircontrol tests tools packaging/pyinstaller_hooks run_app.py
python -m unittest discover -s tests
python -m aircontrol doctor --no-camera
python -m aircontrol selftest
```

## Локальная Сборка

PyInstaller не кросс-компилирует. Windows-сборку нужно делать на Windows,
Linux-сборку на Linux, macOS-сборку на macOS. Для всех трёх ОС используйте
GitHub Actions.

```bash
python tools/check_tracked_sources.py
pip install -r requirements-build.txt
go test ./cmd/aircontrol-helper
mkdir -p bin
go build -trimpath -ldflags="-s -w" -o bin/aircontrol-helper ./cmd/aircontrol-helper
pyinstaller aircontrol.spec --noconfirm
python tools/smoke_build.py
```

На Windows используйте имя `bin/aircontrol-helper.exe`. В GitHub Actions это
делается автоматически на каждой ОС.

Результат:

```text
dist/
  AirControl/
  AirControl.app/        # только macOS
```

## Python Package Metadata

Проект содержит `pyproject.toml`, `MANIFEST.in`, `LICENSE`, `NOTICE` и
`CITATION.cff`. Это нужно для аккуратной установки из исходников, editable-mode,
академического цитирования и корректного source distribution.

Проверка исходного пакета:

```bash
python -m build --sdist --wheel
python -m pip install --force-reinstall dist/aircontrol_assistive-*.whl
aircontrol doctor --no-camera
```

## Упаковка Вручную

macOS:

```bash
cp packaging/USER_GUIDE_RU.txt dist/AirControl/USER_GUIDE_RU.txt
cd dist
zip -r ../AirControl-macOS.zip .
```

Windows:

```powershell
Copy-Item packaging\USER_GUIDE_RU.txt dist\AirControl\USER_GUIDE_RU.txt
Compress-Archive -Path dist\* -DestinationPath AirControl-Windows.zip
& "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging\windows\AirControl.iss
```

Linux:

```bash
sudo apt install python3-tk libgl1 libgles2 libegl1 libglib2.0-0 xdotool xvfb
cp packaging/USER_GUIDE_RU.txt dist/AirControl/USER_GUIDE_RU.txt
cp packaging/linux/AirControl.desktop dist/AirControl/AirControl.desktop
tar -czf AirControl-Linux.tar.gz -C dist AirControl
bash packaging/linux/build_appimage.sh
bash packaging/linux/build_deb.sh
```

## Проверка Артефактов

После упаковки:

```bash
python tools/verify_release_artifacts.py --os macOS
python tools/verify_release_artifacts.py --os Windows
python tools/verify_release_artifacts.py --os Linux --skip-appimage-run
```

Verifier проверяет наличие исполняемого файла, `USER_GUIDE_RU.txt`,
Linux desktop-файла, Windows installer, native helper и состав bundled
FLAC-конвертеров SpeechRecognition.

## Инструкция Для Тестеров

### Windows

1. Запустить `AirControl-Setup.exe`.
2. Открыть AirControl через ярлык на рабочем столе или в меню Пуск.
3. Если SmartScreen ругается на неподписанное приложение: `Подробнее` ->
   `Выполнить в любом случае`.

### macOS

1. Распаковать `AirControl-macOS.zip`.
2. Перенести `AirControl.app` в Applications.
3. Первый запуск: правый клик по приложению -> `Open`.
4. Разрешить Camera, Microphone и Accessibility в System Settings.

### Linux

1. Для Debian/Ubuntu открыть `AirControl-Linux-amd64.deb` графическим
   установщиком.
2. Portable-вариант: скачать `AirControl-Linux-x86_64.AppImage`, включить право
   запуска и открыть двойным кликом.
3. Если камера недоступна: добавить пользователя в группу `video` и
   перелогиниться.
4. Если жесты видны, но курсор не двигается: проверить Xorg/Wayland и
   `ydotoold`, затем сохранить ZIP-отчёт диагностики.

## Release Branch Flow

Рекомендуемый порядок:

```bash
git checkout develop
git checkout -b feature/my-change
# небольшие осмысленные коммиты
git checkout develop
git merge --no-ff feature/my-change
git checkout -b release/v2.x
# версия, документация, packaging polish
git checkout main
git merge --no-ff release/v2.x
git tag -a v2.x.0 -m "AirControl v2.x.0"
git branch -f deploy main
```

Так в истории остаются видны отдельные продуктовые этапы, release preparation и
финальная интеграция в `main`.

## Заметки Для Релиза

- Сборки сейчас не подписаны.
- Размер бандла большой из-за MediaPipe, OpenCV и runtime.
- Research-команды лучше запускать из исходников с `requirements-optional.txt`.
- Пользовательские данные пишутся в пользовательский каталог AirControl и не
  должны попадать в git.
