<#
.SYNOPSIS
    Convenience wrapper for docker compose commands.

.EXAMPLE
    .\make.ps1 up        # build images and start both services
    .\make.ps1 down      # stop and remove containers
    .\make.ps1 logs      # stream logs from all services
    .\make.ps1 rebuild   # force-rebuild images with no cache
#>
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("up", "down", "logs", "rebuild")]
    [string]$Target
)

$compose = @("docker", "compose", "-f", "docker/docker-compose.yml")

switch ($Target) {
    "up"      { & $compose[0] $compose[1..($compose.Length-1)] up --build }
    "down"    { & $compose[0] $compose[1..($compose.Length-1)] down }
    "logs"    { & $compose[0] $compose[1..($compose.Length-1)] logs -f }
    "rebuild" { & $compose[0] $compose[1..($compose.Length-1)] build --no-cache }
}
