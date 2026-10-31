@echo off
setlocal
call "%~dp0conda_windows.bat" update
if errorlevel 1 goto fail
echo Installation complete. Run start_windows.bat.
pause
exit /b 0
:fail
echo Installation failed. Review the error above.
pause
exit /b 1
