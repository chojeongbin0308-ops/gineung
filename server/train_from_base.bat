@echo off
REM === Train a new version from the original ko-sroberta (not on top of the current model) ===
call "%~dp0train_v2.bat" --from-base
