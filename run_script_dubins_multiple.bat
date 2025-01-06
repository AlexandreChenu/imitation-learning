@echo off
setlocal

set CONDA_ENV=E:\alexandre.chenu\envs\imitation_env
set PYTHON_PATH=python
set SCRIPT_PATH=train_GAIL_dubins.py
set RUN_COUNT=10

REM Initialize Conda
call "E:\alexandre.chenu\Scripts\activate.bat" 
call conda activate %CONDA_ENV%

for /L %%i in (1,1,%RUN_COUNT%) do (
    echo Running iteration %%i
    %PYTHON_PATH% %SCRIPT_PATH% 
)

echo Finished running %SCRIPT_PATH% %RUN_COUNT% times
endlocal
pause