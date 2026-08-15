@echo off
REM ===================================================================
REM  Room Mapper - one-click demo
REM
REM  Runs the virtual robot through a room and opens the live map.
REM  No hardware, no broker, no Docker needed.
REM ===================================================================

setlocal
cd /d "%~dp0"

echo.
echo  ================================================
echo   Room Mapper - 3 Wheel Mobile Robot
echo   Universiti Teknologi PETRONAS
echo  ================================================
echo.

REM The Microsoft Store stub named "python" is not a real interpreter, so
REM prefer the py launcher and fall back only if it is missing.
where py >nul 2>&1
if %errorlevel%==0 (
    set "PY=py"
) else (
    where python >nul 2>&1
    if %errorlevel%==0 (
        set "PY=python"
    ) else (
        echo  [ERROR] Python not found.
        echo  Install it from https://python.org and tick "Add to PATH".
        pause
        exit /b 1
    )
)

echo  Checking dependencies...
%PY% -c "import fastapi, uvicorn, numpy, pydantic, websockets" >nul 2>&1
if not %errorlevel%==0 (
    echo  Installing required packages, this may take a minute...
    %PY% -m pip install -r requirements.txt --quiet
    %PY% -m pip install websockets --quiet
)

echo  Dependencies OK.
echo.
echo  Choose a room:
echo    [1] Rectangular  6.0 x 4.5 m   ^(27.0 m2^)
echo    [2] L-shaped                   ^(25.0 m2^)
echo    [3] Rectangular with furniture ^(27.0 m2^)
echo.
set /p ROOMCHOICE="  Enter 1, 2 or 3 (default 1): "

if "%ROOMCHOICE%"=="2" (set ROOM=l-shaped) else (
if "%ROOMCHOICE%"=="3" (set ROOM=furnished) else (
set ROOM=rectangular))

echo.
echo  Mapping the %ROOM% room.
echo  Opening http://localhost:8080 - close this window to stop.
echo.

start "" http://localhost:8080
%PY% services\mapper\main.py --source sim --room %ROOM% --speed 8 --port 8080

pause
