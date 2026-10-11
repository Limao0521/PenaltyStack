@echo off
setlocal
rem Experimento final pareado del portero: FSM base, v2, v3, v4, v5 balon y v5.
rem Semillas nuevas 70001; 50 penales por estilo y portero (8 estilos).
cd /d "%~dp0"
set OUT=runs\keeper_fusion
if not exist "%OUT%" mkdir "%OUT%"
echo === 1/3 Pruebas unitarias ===
".venv\Scripts\python.exe" -m unittest discover -s tests > "%OUT%\tests.log" 2>&1
if errorlevel 1 (echo Algunas pruebas fallaron: ver %OUT%\tests.log) else (echo Pruebas OK)
echo === 2/3 Experimento final: scripted vision pose pose_v4 ball_v5 full_v5 ===
".venv\Scripts\python.exe" -u tools\evaluate_keeper_fusion.py --keepers scripted vision pose pose_v4 ball_v5 full_v5 --episodes 50 --seed 70001 --output "%OUT%\final_70001.json" --log "%OUT%\final_70001.log"
echo === 3/3 Trazas del observador en las mismas semillas (solo analisis) ===
".venv\Scripts\python.exe" -u tools\evaluate_keeper_fusion.py --keepers observer --episodes 50 --seed 70001 --frames --output "%OUT%\observer_70001.json" --log "%OUT%\observer_70001.log"
echo Terminado. Resultados en %OUT%
pause
