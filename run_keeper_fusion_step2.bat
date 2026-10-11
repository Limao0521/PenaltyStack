@echo off
setlocal
rem Solo el paso 2: duelo pareado de seis porteros (semillas 70001, 50 por estilo).
rem Reintenta trabajos si un proceso muere y guarda cualquier error en el log.
cd /d "%~dp0"
set OUT=runs\keeper_fusion
if not exist "%OUT%" mkdir "%OUT%"
echo === Experimento final: scripted vision pose pose_v4 ball_v5 full_v5 ===
".venv\Scripts\python.exe" -u tools\evaluate_keeper_fusion.py --keepers scripted vision pose pose_v4 ball_v5 full_v5 --episodes 50 --seed 70001 --output "%OUT%\final_70001.json" --log "%OUT%\final_70001.log" 2> "%OUT%\final_70001.stderr.log"
echo Terminado. Resultados en %OUT%
pause
