@echo off
title Morphic-CV Building Room Detector
cd /d "%~dp0"
echo Starting Morphic-CV Room Detector...
python predict_room.py
pause
