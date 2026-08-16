@echo off
REM ===================================================================
REM  Stop everything START-TWIN.bat started.
REM
REM  Omniverse is closed too. It holds the GPU at full tilt while a
REM  scene is open, which on a laptop is the difference between warm
REM  and uncomfortable.
REM ===================================================================

setlocal
cd /d "%~dp0"

echo.
echo  Stopping the digital twin...
echo.

echo  [1/2] Closing Omniverse...
tasklist /fi "imagename eq kit.exe" 2>nul | find /i "kit.exe" >nul
if errorlevel 1 (
    echo        not running.
) else (
    taskkill /f /im kit.exe >nul 2>&1
    echo        closed.
)

echo  [2/2] Stopping the stack...
docker compose down >nul 2>&1
if errorlevel 1 (
    echo        [WARN] docker compose down reported a problem.
    echo        Check with:  docker compose ps
) else (
    echo        stopped.
)

echo.
echo  All stopped. Scans you saved are still in .\scans and the
echo  measurement history is still in InfluxDB.
echo.
pause
