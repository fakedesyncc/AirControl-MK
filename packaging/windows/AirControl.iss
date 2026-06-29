#define MyAppName "AirControl"
#define MyAppVersion "2.2.0"
#define MyAppPublisher "AirControl"
#define MyAppExeName "AirControl.exe"

[Setup]
AppId={{4F69F4F1-8606-4C63-93A9-A8D8218D7F1C}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=..\..\installer
OutputBaseFilename=AirControl-Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest

[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"

[Tasks]
Name: "desktopicon"; Description: "Создать ярлык на рабочем столе"; GroupDescription: "Ярлыки:"; Flags: unchecked

[Files]
Source: "..\..\dist\AirControl\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\..\packaging\USER_GUIDE_RU.txt"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\AirControl"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Быстрый старт AirControl"; Filename: "{app}\USER_GUIDE_RU.txt"
Name: "{autodesktop}\AirControl"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Запустить AirControl"; Flags: nowait postinstall skipifsilent
