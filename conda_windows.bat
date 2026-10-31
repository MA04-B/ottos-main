@echo off
rem Called from Windows entry points; does not localize environment changes.
cd /d "%~dp0"
if errorlevel 1 exit /b 1
set "CONDA_CMD="
for /f "delims=" %%C in ('where conda.bat 2^>nul') do if not defined CONDA_CMD set "CONDA_CMD=%%C"
if not defined CONDA_CMD if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" set "CONDA_CMD=%USERPROFILE%\miniconda3\condabin\conda.bat"
if not defined CONDA_CMD if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" set "CONDA_CMD=%USERPROFILE%\anaconda3\condabin\conda.bat"
if not defined CONDA_CMD if exist "%USERPROFILE%\miniforge3\condabin\conda.bat" set "CONDA_CMD=%USERPROFILE%\miniforge3\condabin\conda.bat"
if not defined CONDA_CMD if exist "%LOCALAPPDATA%\miniconda3\condabin\conda.bat" set "CONDA_CMD=%LOCALAPPDATA%\miniconda3\condabin\conda.bat"
if not defined CONDA_CMD if exist "%LOCALAPPDATA%\anaconda3\condabin\conda.bat" set "CONDA_CMD=%LOCALAPPDATA%\anaconda3\condabin\conda.bat"
if not defined CONDA_CMD if exist "%ProgramData%\miniconda3\condabin\conda.bat" set "CONDA_CMD=%ProgramData%\miniconda3\condabin\conda.bat"
if not defined CONDA_CMD if exist "%ProgramData%\anaconda3\condabin\conda.bat" set "CONDA_CMD=%ProgramData%\anaconda3\condabin\conda.bat"
if not defined CONDA_CMD (
    echo Conda not found. Install Miniconda from https://docs.conda.io/projects/miniconda/
    echo Then run this script from Anaconda Prompt or Miniconda Prompt.
    exit /b 1
)
if /i "%~1"=="update" goto update
call "%CONDA_CMD%" activate otoslotto-ai
if errorlevel 1 exit /b 1
exit /b 0

:update
set "ENV_LIST=%TEMP%\otoslotto-envs-%RANDOM%-%RANDOM%.txt"
call "%CONDA_CMD%" env list > "%ENV_LIST%"
if errorlevel 1 (
    del "%ENV_LIST%" 2>nul
    echo Conda could not enumerate environments. Review the error above.
    exit /b 1
)
findstr /r /c:"^otoslotto-ai " "%ENV_LIST%" >nul
if errorlevel 1 (
    call "%CONDA_CMD%" env create -f "%~dp0environment.yml" -y
) else (
    call "%CONDA_CMD%" env update -n otoslotto-ai -f "%~dp0environment.yml" --prune
)
set "ENV_UPDATE_EXIT=%ERRORLEVEL%"
del "%ENV_LIST%" 2>nul
if not "%ENV_UPDATE_EXIT%"=="0" exit /b 1
call "%CONDA_CMD%" activate otoslotto-ai
if errorlevel 1 exit /b 1
python -m src.cli verify
if errorlevel 1 exit /b 1
python -m src.cli init
if errorlevel 1 exit /b 1
exit /b 0
