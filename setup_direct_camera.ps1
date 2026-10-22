# Route the Dahua camera that is cabled straight into this PC's Ethernet port.
#
# The camera keeps its address 192.168.29.215, but the Wi-Fi is on the same 192.168.29.x network,
# so Windows would send camera traffic out through Wi-Fi. This gives the Ethernet port one extra
# address (192.168.29.250, as a /32 so the rest of 192.168.29.x stays on Wi-Fi) and routes only the
# camera's address through the cable. The Ethernet port keeps its 192.168.1.1 address.
#
# Run once in an administrator PowerShell:   powershell -ExecutionPolicy Bypass -File setup_direct_camera.ps1
# Undo:   Remove-NetRoute -DestinationPrefix 192.168.29.215/32 -Confirm:$false
#         Remove-NetIPAddress -IPAddress 192.168.29.250 -Confirm:$false

$Camera = "192.168.29.215"
$PcAddress = "192.168.29.250"
$Adapter = "Ethernet"

if (-not (Get-NetIPAddress -IPAddress $PcAddress -ErrorAction SilentlyContinue)) {
    New-NetIPAddress -InterfaceAlias $Adapter -IPAddress $PcAddress -PrefixLength 32 | Out-Null
    "Added $PcAddress/32 to $Adapter"
}
if (-not (Get-NetRoute -DestinationPrefix "$Camera/32" -ErrorAction SilentlyContinue)) {
    New-NetRoute -DestinationPrefix "$Camera/32" -InterfaceAlias $Adapter -NextHop 0.0.0.0 | Out-Null
    "Routed $Camera through $Adapter"
}

$client = New-Object System.Net.Sockets.TcpClient
$ok = $client.ConnectAsync($Camera, 554).Wait(3000)
$client.Close()
if ($ok) { "Camera $Camera answers on RTSP port 554 - start Jewel Guard (Direct Ethernet)." }
else { "Camera $Camera still not reachable: check that its IP really is $Camera and that it is powered." }
