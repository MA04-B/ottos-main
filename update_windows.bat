@echo off
setlocal
call "%~dp0conda_windows.bat" update
if errorlevel 1 goto fail
python -m src.cli sync
if errorlevel 1 goto fail
echo Environment and historical database updated.
pause
exit /b 0
:fail
echo Update failed. Existing valid historical data is preserved.
pause
exit /b 1
