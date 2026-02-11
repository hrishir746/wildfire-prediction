# OpenSSH Server Setup Script
# Right-click this file and select "Run with PowerShell as Administrator"

Write-Host "=== Installing OpenSSH Server ===" -ForegroundColor Cyan
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0

Write-Host "`n=== Starting SSH Service ===" -ForegroundColor Cyan
Start-Service sshd
Set-Service -Name sshd -StartupType Automatic

Write-Host "`n=== Verifying Service Status ===" -ForegroundColor Cyan
Get-Service sshd

Write-Host "`n=== Adding Firewall Rule ===" -ForegroundColor Cyan
New-NetFirewallRule -Name sshd -DisplayName "OpenSSH Server" -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22 -ErrorAction SilentlyContinue

Write-Host "`n=== Your IP Addresses ===" -ForegroundColor Cyan
ipconfig | Select-String "IPv4"

Write-Host "`n=== Setup Complete! ===" -ForegroundColor Green
Write-Host "To test locally, run: ssh $env:USERNAME@127.0.0.1" -ForegroundColor Yellow
Write-Host "`nPress any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
