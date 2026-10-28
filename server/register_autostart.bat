@echo off
REM === Start the server automatically when this user logs in ===
schtasks /create /f /tn "GineungAIServer" /sc onlogon /rl limited /tr "\"%~dp0start_server.bat\""
echo Registered. To remove: schtasks /delete /tn GineungAIServer /f
pause
