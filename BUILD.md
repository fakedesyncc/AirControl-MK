# Сборка AirControl в исполняемый файл

Приложение упаковывается в standalone-бандл (со всеми зависимостями, включая
Python runtime, MediaPipe и модель) через **PyInstaller**. Пользователю не нужен
Python, Node.js, компилятор или другой язык программирования — он запускает
готовый установщик/приложение.

> ⚠️ PyInstaller **не умеет кросс-компиляцию**: бинарник под Windows собирается
> на Windows, под Linux — на Linux, под macOS — на macOS. Поэтому для всех трёх
> ОС используйте **GitHub Actions** (раздел ниже) — это самый простой способ
> получить файлы для тестеров на всех платформах сразу.

---

## Вариант 1 — автосборка под 3 ОС через GitHub Actions (рекомендуется)

1. Залейте проект в репозиторий GitHub.
2. Вкладка **Actions** → workflow **«Build AirControl»** → **Run workflow**
   (или запушьте тег: `git tag v1.0 && git push --tags`).
3. По завершении скачайте артефакты:
   - `AirControl-macOS.zip`
   - `AirControl-Windows.zip`
   - `AirControl-Setup.exe`
   - `AirControl-Linux-amd64.deb`
   - `AirControl-Linux-x86_64.AppImage`
   - `AirControl-Linux.tar.gz`
4. Для обычных пользователей раздавайте `AirControl-Setup.exe` на Windows,
   `AirControl-macOS.zip` на macOS, `AirControl-Linux-amd64.deb` на Debian/Ubuntu
   и `AirControl-Linux-x86_64.AppImage` как portable-вариант для Linux.
   `AirControl-Linux.tar.gz` оставьте для диагностики и ручного теста.

Файл workflow: `.github/workflows/build.yml`.

---

## Вариант 2 — локальная сборка (только под текущую ОС)

```bash
pip install -r requirements-build.txt
python -m compileall aircontrol tests tools packaging/pyinstaller_hooks run_app.py
python -m unittest discover -s tests
pyinstaller aircontrol.spec --noconfirm
python tools/smoke_build.py
```

`requirements-build.txt` ставит только стабильное ядро для пользовательского
бандла. Голосовой микрофон, `report` и расширенные ML-бэкенды находятся в
`requirements-optional.txt`; их лучше ставить только для разработки/исследований.

Результат в `dist/`:
- **macOS**: `dist/AirControl.app` (и папка `dist/AirControl/`)
- **Windows**: `dist/AirControl/AirControl.exe`
- **Linux**: `dist/AirControl/AirControl`

Упаковать для передачи:
```bash
cp packaging/USER_GUIDE_RU.txt dist/AirControl/USER_GUIDE_RU.txt
cd dist && zip -r ../AirControl-macOS.zip .        # macOS
# Windows: Compress-Archive -Path dist/* -DestinationPath AirControl.zip
# Linux лучше упаковывать tar.gz, чтобы сохранить права запуска.
```

Для локальной проверки скачиваемого архива после упаковки, из корня проекта:
```bash
python tools/verify_release_artifacts.py --os macOS   # Windows/Linux: соответствующая ОС
```
Этот verifier проверяет не только наличие `AirControl` и руководства, но и
release-инварианты вроде состава bundled FLAC-конвертеров SpeechRecognition.

---

## Инструкция для тестеров

**macOS**
1. Распаковать архив, перенести `AirControl.app` в Программы.
2. Первый запуск: ПКМ → «Открыть» (обход Gatekeeper для неподписанного приложения).
3. Разрешить доступ к **камере**, **микрофону** и **универсальному доступу**
   (Системные настройки → Конфиденциальность и безопасность). Перезапустить.
4. Запустить `AirControl.app` двойным кликом и выбрать безопасную тренировку
   или калибровку. Калибровка работает в Tk-окне с кнопками и не требует
   Space/Esc.

**Windows**
1. Для обычного пользователя: запустить `AirControl-Setup.exe`, затем ярлык
   AirControl на рабочем столе или в меню Пуск. Установщик ставит приложение
   в профиль пользователя и не требует прав администратора.
2. Для portable-теста: распаковать `AirControl-Windows.zip` и запустить
   `AirControl.exe`.
3. SmartScreen → «Подробнее» → «Выполнить в любом случае» (приложение не подписано).

**Linux**
1. На Debian/Ubuntu скачать `AirControl-Linux-amd64.deb` и открыть его двойным
   кликом через графический установщик. После установки AirControl появится в
   меню приложений.
2. Portable-вариант без установки: скачать `AirControl-Linux-x86_64.AppImage`.
3. Если AppImage не запускается двойным кликом, один раз включить право запуска:
   `chmod +x AirControl-Linux-x86_64.AppImage`.
4. Запустить AppImage двойным кликом. Portable-архив: распаковать
   `AirControl-Linux.tar.gz` и запустить файл `AirControl`.
5. Если нет доступа к камере: `sudo usermod -a -G video $USER` и перелогиниться.
6. Может понадобиться: `sudo apt install libgl1 libglib2.0-0 xdotool`.
   Для Wayland-теста управления дополнительно можно настроить `ydotool` и
   `ydotoold` с доступом к `/dev/uinput`.
7. Для диагностики запустить из терминала:
   `./AirControl doctor` или `python -m aircontrol doctor`.
   Для безопасной проверки реального движения курсора без кликов:
   `./AirControl doctor --input-probe`.
8. Для безопасной ассистивной настройки сначала запустить:
   `./AirControl assistive --dry-input`, затем `./AirControl assistive`.

В обычном GUI-сценарии пользователь может нажать «Калибровка под пользователя»:
окно показывает видеопревью, крупные кнопки и шаги активной зоны/щипка без
обязательного использования клавиатуры.
Кнопка «Проверить систему» показывает `doctor`-диагностику без терминала, а
«Просмотр камеры» всегда запускается в безопасном `view`-режиме с отключённым
низкоуровневым вводом.

Заметки по Linux:
- В Wayland-сессии глобальное управление курсором/клавиатурой через X11-бэкенды
  часто ограничено. AirControl пробует `ydotool`, если `ydotoold` уже запущен
  и имеет доступ к `/dev/uinput`; иначе для проверки режима управления
  используйте Xorg-сессию.
- Если в окне камеры или `doctor-summary.txt` указано `INPUT RISK`, камера и
  распознавание могут работать, но текущая сессия ещё не гарантирует клики и
  клавиши. Это не ошибка жеста: проверьте Xorg/Wayland и `ydotoold`.
- Если `pynput` недоступен, в Xorg-сессии приложение пробует fallback через
  `xdotool`.
- Для громкости используются `wpctl`, `pactl` или `amixer`.
- Для скриншотов используются доступные системные утилиты: `grim`,
  `gnome-screenshot`, `spectacle`, `scrot`, `import`, затем фолбэк `mss`.
- Для сворачивания чужих окон нужны `wmctrl` и `xdotool`.

Пользовательские данные (конфиг, датасеты, логи, скриншоты) сохраняются в
`~/.aircontrol/`.

---

## Заметки

- Приложение собрано без подписи. Для распространения вне тестов нужна подпись
  и нотаризация (Apple Developer ID) / подпись Authenticode (Windows).
- Размер бандла большой (~270–350 МБ) из-за MediaPipe, OpenCV и голосовых
  библиотек. Research-команды (`report`, `train --backend rf/mlp`) лучше
  запускать из исходников с полным Python-окружением; пользовательский бандл
  оптимизирован под GUI, жесты, камеру, диагностику и ассистивное управление.
- Другие языки программирования не должны быть требованием для пользователя.
  Текущая схема оставляет Python внутри собранного приложения, а внешние
  технологии использует только для упаковки: Inno Setup на Windows, Debian
  package tools на Linux и PyInstaller `.app` на macOS. Если позже понадобится
  автообновление, системный tray или сервисный демон ввода, это лучше добавлять
  как тонкую оболочку на Rust/Tauri или платформенных API, не переписывая
  распознавание жестов.
- Сборка идёт в режиме `console=False` (без окна терминала). Чтобы видеть логи
  при отладке — временно поставьте `console=True` в `aircontrol.spec`.
