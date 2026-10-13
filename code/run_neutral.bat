@echo off
rem 中性臂 LLM 批启动器：实际工作由 run_neutral.ps1 完成（12 窗口 = 4 方法 x 3 种子）
rem 中断后双击本文件即可断点续跑
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_neutral.ps1"
pause
