@echo off

rem 激活conda环境
call "G:\Anaconda\Scripts\activate.bat" zed

rem 运行提取脚本
python extract_svo_frames.py --svo "C:\Users\ZYF\Documents\ZED\HD1080_SN36245620_11-13-56.svo2" --max-frames 20

rem 暂停以便查看输出
pause
