@echo off
rem PenaltyStack: opens the deploy dialog for the Copa NAO CCM MX 2026 penalty robots.
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
  start "" ".venv\Scripts\pythonw.exe" deploy\deploy_dialog.py
) else (
  start "" pythonw deploy\deploy_dialog.py
)
