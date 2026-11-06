@echo off
setlocal
cd /d "%~dp0"

echo ========================================
echo   Employee Ticket Routing
echo ========================================
echo.

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start.ps1"
if errorlevel 1 (
    echo.
    echo Startup failed. Review the error above and try again.
    pause
)

endlocal
