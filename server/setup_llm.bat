@echo off
REM === Gineung RAG: check Ollama and download the Korean LLM (once, about 5GB) ===
cd /d "%~dp0"
where ollama >nul 2>nul || (echo [!] Ollama is not installed. Install it from https://ollama.com/download  then run this again. & pause & exit /b 1)
echo Ollama found:
ollama --version
set MODEL=exaone3.5:7.8b
if not "%1"=="" set MODEL=%1
echo Downloading %MODEL% (first time only, a few minutes)...
ollama pull %MODEL%
echo.
echo Installed models:
ollama list
ollama list > llm_check.txt 2>&1
echo.
echo Quick test:
ollama run %MODEL% "임실군 주민에게 인사 한 문장" 
echo.
echo Done. Restart start_server.bat - the server will use this model for RAG answers.
pause
