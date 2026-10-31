@echo off
setlocal
call "%~dp0conda_windows.bat"
if errorlevel 1 goto fail
python -m src.cli verify
if errorlevel 1 goto fail
echo Open http://127.0.0.1:8501
python -m streamlit run "%~dp0app.py" --server.address 127.0.0.1 --server.port 8501
if errorlevel 1 goto fail
pause
exit /b 0
:fail
echo Startup failed. Run install_windows.bat and review the error above.
pause
exit /b 1
