param(
  [Parameter(Mandatory=$true)][string]$Server,
  [Parameter(Mandatory=$true)][string]$User,
  [int]$Port = 22,
  [string]$KeyFile = "",
  [string]$RemoteRoot = "/root/autodl-tmp/ssgf",
  [string]$LocalDir = ".\server_results"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Path $LocalDir -Force | Out-Null
$target = "$User@$Server"
$sshArgs = @("-P", "$Port")
$scpArgs = @("-P", "$Port", "-r")
if ($KeyFile) {
  $sshArgs += @("-i", $KeyFile)
  $scpArgs += @("-i", $KeyFile)
}

& ssh @sshArgs $target "cd '$RemoteRoot' && tar -czf /tmp/ssgf_results.tar.gz results review paper"
& scp @scpArgs "${target}:/tmp/ssgf_results.tar.gz" $LocalDir
Get-FileHash "$LocalDir\ssgf_results.tar.gz" -Algorithm SHA256
