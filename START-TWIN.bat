@echo off
REM ===================================================================
REM  Digital Twin - one click, all three windows
REM
REM   1. Grafana        http://localhost:3000   the data
REM   2. Web app        http://localhost:8080   the robot's own 2D map
REM   3. Omniverse                              the physical world
REM
REM  Omniverse is the world here, not a picture of it: drag a table in
REM  the viewport and the robot bumps into it where you put it, and the
REM  contact appears on the 2D map.
REM
REM  Needs Docker Desktop running. Omniverse is optional - the other two
REM  windows work without it and this says so rather than failing.
REM
REM  Universiti Teknologi PETRONAS - 3 Wheel Holonomic Mobile Robot
REM ===================================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo  ===============================================================
echo   Digital Twin  -  3 Wheel Holonomic Mobile Robot
echo  ===============================================================
echo.

REM -- 1. Docker ------------------------------------------------------
docker version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Docker is not running.
    echo.
    echo  Start Docker Desktop, wait for the whale icon to stop
    echo  animating, then run this again.
    echo.
    pause
    exit /b 1
)
echo  [1/5] Docker is running.

REM -- 2. The stack ---------------------------------------------------
echo  [2/5] Starting the stack ^(mapper, Grafana, InfluxDB, MQTT^)...
docker compose up -d --build >nul 2>&1
if errorlevel 1 (
    echo.
    echo  [ERROR] docker compose failed. The output was hidden to keep
    echo  this readable - run it directly to see why:
    echo.
    echo      docker compose up -d --build
    echo.
    pause
    exit /b 1
)

REM -- 3. Wait for the mapper -----------------------------------------
REM  Waited on properly rather than with a fixed sleep: a rebuild can
REM  take a couple of minutes and a fixed wait either wastes time or
REM  opens a browser at a page that is not there yet.
echo  [3/5] Waiting for the mapper to come up...
set MAPPER=
for /l %%i in (1,1,60) do (
    if not defined MAPPER (
        curl -s -o nul -m 3 http://127.0.0.1:8080/health >nul 2>&1
        if not errorlevel 1 (
            set MAPPER=yes
            echo        ready after about %%i seconds.
        ) else (
            REM  ping rather than timeout: `timeout` refuses to run when
            REM  stdin is redirected, which is exactly what happens when
            REM  this file is driven from a script or a CI step.
            ping -n 2 127.0.0.1 >nul
        )
    )
)
if not defined MAPPER (
    echo.
    echo  [ERROR] The mapper did not answer within 60 seconds.
    echo  Check what it said:   docker logs roommapper-mapper
    echo.
    pause
    exit /b 1
)

REM -- 4. Omniverse ---------------------------------------------------
REM  Kit's --exec cannot open a path containing spaces, and this project
REM  lives under "00 Reseach Project". So the scene is copied somewhere
REM  without any, and Kit is pointed at the copy.
echo  [4/5] Starting Omniverse...

set "KIT="
if defined OMNIVERSE_KIT (
    if exist "%OMNIVERSE_KIT%" set "KIT=%OMNIVERSE_KIT%"
)
if not defined KIT (
    for %%p in (
        "C:\Omniverse\kit-app-template\_build\windows-x86_64\release\digital_twin_viewer.kit.bat"
        "C:\Omniverse\kit-app-template\_build\windows-x86_64\release\kit.bat"
    ) do (
        if not defined KIT if exist %%p set "KIT=%%~p"
    )
)

if not defined KIT (
    echo        Omniverse not found - skipping the 3D window.
    echo.
    echo        The other two windows work without it. To include it,
    echo        set OMNIVERSE_KIT to your .kit.bat, for example:
    echo.
    echo          set OMNIVERSE_KIT=C:\Omniverse\kit-app-template\_build\windows-x86_64\release\digital_twin_viewer.kit.bat
    echo.
) else (
    if not exist "C:\kitscene" mkdir "C:\kitscene" >nul 2>&1
    copy /y "%~dp0omniverse\kit_room_3d.py" "C:\kitscene\kit_room_3d.py" >nul
    if errorlevel 1 (
        echo        [WARN] Could not stage the scene into C:\kitscene - skipping 3D.
    ) else (
        REM  Launched through PowerShell rather than `start`.
        REM
        REM  `start "" "some.bat" --exec "path"` returns success and then
        REM  nothing runs: cmd mangles the quoting around a .bat target that
        REM  itself takes quoted arguments, and the child dies silently, which
        REM  is the worst way for this to fail - the launcher says Omniverse is
        REM  coming up and no window ever appears. Start-Process takes the
        REM  executable and its arguments as separate values, so there is no
        REM  quoting for anything to mangle.
        powershell -NoProfile -Command "Start-Process -FilePath '!KIT!' -ArgumentList '--exec','C:/kitscene/kit_room_3d.py'"
        echo        launching - the RTX renderer takes about a minute.
    )
)

REM -- 5. The two browser windows -------------------------------------
echo  [5/5] Opening the dashboards...
start "" http://localhost:8080
start "" http://localhost:3000

echo.
echo  ===============================================================
echo   Running.
echo.
echo     Web app     http://localhost:8080   the robot's 2D map
echo     Grafana     http://localhost:3000   login admin / admin
echo     Omniverse   the 3D window, once the renderer finishes
echo.
echo   Try this:
echo     - drag a table in the Omniverse viewport
echo     - press "Scan again" on the web app
echo     - watch the robot find it, and the obstacle appear on the map
echo.
echo   To stop everything:   STOP-TWIN.bat
echo  ===============================================================
echo.
pause
