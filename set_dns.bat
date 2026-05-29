@echo off
:: Check for administrative privileges
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] This script must be run as an Administrator.
    echo Please right-click this file and select "Run as administrator".
    pause
    exit /b 1
)

echo Setting IPv4 DNS to 1.1.1.3 for Wi-Fi...
netsh interface ip set dns name="Wi-Fi" static 1.1.1.3

echo Setting IPv6 DNS to 2606:4700:4700::1113 for Wi-Fi...
netsh interface ipv6 set dns name="Wi-Fi" static 2606:4700:4700::1113

echo.
echo DNS settings have been updated.
ipconfig /flushdns
echo.
pause
