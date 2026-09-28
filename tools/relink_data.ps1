<#
  relink_data.ps1 —— 按 <仓库根>\paths.json 把数据集目录搬到外接盘，并建立/维护目录联接（junction）

  为什么用 junction（目录联接）：
    · Windows 自带，**不需要管理员权限**（`mklink /J` 与 `mklink /D` 不同）
    · 仓库内所有相对路径（main\001\...、l1\L1-03\...）**照旧有效** ⇒ 代码零改动
    · 拔盘时目录会变成"空"，插回来自动恢复；换盘只改 paths.json 再跑一次 -Apply

  四种模式（默认只预览，什么都不改）：
    （无开关）   预览：列出每条映射的 仓库侧状态 / 目标侧状态
    -Migrate     搬迁：把仓库里的真实目录搬到目标位，再在该位置建立 junction
    -Apply       重接：数据已在目标位时，建立/更新 junction（目标缺目录会自动建空目录）
    -Unlink      拆链：只移除 junction，**绝不动**外接盘上的数据
    -Restore     还原：必须与 -Unlink 同时用 —— 拆链后把数据从外接盘搬回仓库

  例：
    powershell -ExecutionPolicy Bypass -File tools\relink_data.ps1
    powershell -ExecutionPolicy Bypass -File tools\relink_data.ps1 -Migrate
    powershell -ExecutionPolicy Bypass -File tools\relink_data.ps1 -Unlink -Restore
#>
[CmdletBinding()]
param(
    [switch]$Apply,
    [switch]$Migrate,
    [switch]$Unlink,
    [switch]$Restore,
    [switch]$Force,
    [string]$Config = ''
)

$ErrorActionPreference = 'Stop'
$OutputEncoding = [Console]::OutputEncoding = [Text.Encoding]::UTF8

$Repo = Split-Path -Parent $PSScriptRoot
if (-not $Config) { $Config = Join-Path $Repo 'paths.json' }

if (-not (Test-Path -LiteralPath $Config)) { throw "找不到配置文件：$Config" }
$cfg = Get-Content -LiteralPath $Config -Raw -Encoding UTF8 | ConvertFrom-Json

# 环境变量优先（临时换盘用，不改文件）
$Root = $env:SHM_DATA_ROOT
if ([string]::IsNullOrWhiteSpace($Root)) { $Root = $cfg.data_root }
if ($Root) {
    $Root = $Root.TrimEnd('\', '/')
    # 裸盘符（如 E:）必须补分隔符，否则 Join-Path 会得到盘符相对路径 E:l1
    if ($Root -match '^[A-Za-z]:$') { $Root = $Root + '\' }
}

Write-Host "仓库根   : $Repo"
Write-Host ("数据根   : {0}" -f $(if ($Root) { $Root } else { '(未配置)' }))
Write-Host "配置文件 : $Config"

# ---------------------------------------------------------------- 工具函数
function Expand-Links([string]$rel) {
    $full = Join-Path $Repo ($rel -replace '/', '\')
    if ($rel -match '[*?\[]') {
        return @(Get-ChildItem -Path $full -Directory -Force -ErrorAction SilentlyContinue |
                 ForEach-Object { $_.FullName } | Sort-Object)
    }
    return @($full)
}

function Get-TargetOf($item, [string]$link) {
    $t = $item.target -replace '/', '\'
    $base = if ([string]::IsNullOrWhiteSpace($Root)) { $t } else { Join-Path $Root $t }
    if ($item.link -match '[*?\[]') { $base = Join-Path $base (Split-Path -Leaf $link) }
    return $base
}

function Get-State([string]$p) {
    if (-not (Test-Path -LiteralPath $p)) { return 'missing' }
    $it = Get-Item -LiteralPath $p -Force
    if ($it.Attributes -band [IO.FileAttributes]::ReparsePoint) { return 'junction' }
    if ($it.PSIsContainer) { return 'real' }
    return 'file'
}

function New-Junction([string]$link, [string]$target) {
    if (-not (Test-Path -LiteralPath $target)) {
        $null = New-Item -ItemType Directory -Path $target -Force
    }
    $null = cmd /c ('mklink /J "{0}" "{1}"' -f $link, $target)
    if ($LASTEXITCODE -ne 0) { throw "mklink 建立联接失败：$link -> $target" }
}

function Remove-Junction([string]$link) {
    # 关键：用 rmdir（不带 /s）只删联接本身，绝不递归进目标盘
    $null = cmd /c ('rmdir "{0}"' -f $link)
    if ($LASTEXITCODE -ne 0) { throw "移除联接失败：$link" }
}

function Move-Tree([string]$src, [string]$dst, [string]$what) {
    $parent = Split-Path -Parent $dst
    if ($parent -and -not (Test-Path -LiteralPath $parent)) {
        $null = New-Item -ItemType Directory -Path $parent -Force
    }
    $null = robocopy "$src" "$dst" /E /MOVE /R:1 /W:1 /NFL /NDL /NJH /NJS /NP
    if ($LASTEXITCODE -ge 8) { throw "robocopy $what 失败（退出码 $LASTEXITCODE）：$src -> $dst" }
    if (Test-Path -LiteralPath $src) { Remove-Item -LiteralPath $src -Recurse -Force }
}

function Copy-Tree([string]$src, [string]$dst) {
    $null = robocopy "$src" "$dst" /E /R:1 /W:1 /NFL /NDL /NJH /NJS /NP
    if ($LASTEXITCODE -ge 8) { throw "robocopy 失败（退出码 $LASTEXITCODE）：$src -> $dst" }
}

function Format-MB([long]$bytes) { return ('{0,9:N1} MB' -f ($bytes / 1MB)) }

function Get-DirSize([string]$p) {
    if (-not (Test-Path -LiteralPath $p)) { return 0 }
    return (Get-ChildItem -LiteralPath $p -Recurse -File -Force -ErrorAction SilentlyContinue |
            Measure-Object -Sum Length).Sum
}

function Rel([string]$p) {
    if ($p.StartsWith($Repo, [StringComparison]::OrdinalIgnoreCase)) {
        return $p.Substring($Repo.Length).TrimStart('\')
    }
    return $p
}

# ---------------------------------------------------------------- 展开映射
$rows = @()
foreach ($it in $cfg.items) {
    foreach ($lp in (Expand-Links $it.link)) {
        $tp = Get-TargetOf $it $lp
        $rows += [pscustomobject]@{
            name   = $it.name
            rel    = Rel $lp
            link   = $lp
            target = $tp
            state  = Get-State $lp
            tstate = Get-State $tp
        }
    }
}

if ($rows.Count -eq 0) { Write-Warning 'paths.json 的 items 没有匹配到任何目录。'; return }

# ---------------------------------------------------------------- 预览
$w = ($rows | ForEach-Object { $_.rel.Length } | Measure-Object -Maximum).Maximum
$fmt = '{0,-' + $w + '}  {1,-9}  {2,-6}  {3}'
Write-Host ''
Write-Host ($fmt -f '仓库内路径', '仓库侧', '数据侧', '外接盘目标')
foreach ($r in $rows) {
    $mark = switch ($r.state) { 'junction' { '已联接' } 'real' { '实体目录' } default { '不存在' } }
    $tmark = if ($r.tstate -eq 'missing') { '缺' } else { 'OK' }
    Write-Host ($fmt -f $r.rel, $mark, $tmark, $r.target)
}

$nLink = @($rows | Where-Object { $_.state -eq 'junction' }).Count
$nReal = @($rows | Where-Object { $_.state -eq 'real' }).Count
$nNone = $rows.Count - $nLink - $nReal
Write-Host ''
Write-Host ("合计 {0} 条：已联接 {1} · 仍是本地实体目录 {2} · 不存在 {3}" -f $rows.Count, $nLink, $nReal, $nNone)

# ---------------------------------------------------------------- 各模式
$doMigrate = [bool]$Migrate
$doApply = [bool]$Apply -or $doMigrate
$doUnlink = [bool]$Unlink

if (-not ($doMigrate -or $doApply -or $doUnlink)) {
    Write-Host ''
    Write-Host '（以上仅为预览，未做任何改动）' -ForegroundColor Yellow
    if ($nReal -gt 0) { Write-Host '  要把实体目录搬到外接盘并建联接：加 -Migrate' -ForegroundColor Yellow }
    if ($nReal -eq 0 -and $nNone -gt 0) { Write-Host '  要在空位建立联接：加 -Apply' -ForegroundColor Yellow }
    if ($nLink -gt 0) { Write-Host '  要拆掉联接：加 -Unlink（还原数据再加 -Restore）' -ForegroundColor Yellow }
    return
}

if ($doUnlink) {
    Write-Host ''
    Write-Host '=== 拆链（-Unlink）===' -ForegroundColor Cyan
    foreach ($r in $rows) {
        $rel = $r.rel
        if ($r.state -ne 'junction') { Write-Host ("  跳过 {0}（当前为 {1}）" -f $rel, $r.state); continue }
        Remove-Junction $r.link
        Write-Host ("  已拆 {0}" -f $rel) -ForegroundColor Green
        if ($Restore) {
            $n = (Get-ChildItem -LiteralPath $r.target -Recurse -File -Force -ErrorAction SilentlyContinue |
                  Measure-Object).Count
            Write-Host ("    还原 {0} 个文件 ← {1}" -f $n, $r.target)
            Copy-Tree $r.target $r.link
            if ($Force) {
                Remove-Item -LiteralPath $r.target -Recurse -Force
                Write-Host '    已删除外接盘副本（-Force）' -ForegroundColor Yellow
            } else {
                Write-Host '    外接盘副本保留（要删掉请追加 -Force）' -ForegroundColor Yellow
            }
        }
    }
    Write-Host '拆链完成。' -ForegroundColor Green
    return
}

if ($doMigrate) {
    Write-Host ''
    Write-Host '=== 搬迁（-Migrate）===' -ForegroundColor Cyan
    $bytes = 0
    foreach ($r in $rows) {
        $rel = $r.rel
        if ($r.state -ne 'real') {
            Write-Host ("  跳过 {0}（当前为 {1}，只有实体目录才需要搬迁）" -f $rel, $r.state)
            continue
        }
        if ($r.tstate -ne 'missing') {
            $tgtFiles = (Get-ChildItem -LiteralPath $r.target -Force -ErrorAction SilentlyContinue | Measure-Object).Count
            if ($tgtFiles -gt 0 -and -not $Force) {
                Write-Host ("  跳过 {0}：目标已存在且有内容 → {1}" -f $rel, $r.target) -ForegroundColor Red
                Write-Host '       确认要合并/覆盖，请追加 -Force（会合并同名文件）' -ForegroundColor Red
                continue
            }
        }
        $sz = Get-DirSize $r.link
        Write-Host ("  搬迁 {0}  {1} → {2}" -f $rel, (Format-MB $sz), $r.target)
        Move-Tree $r.link $r.target '搬迁'
        New-Junction $r.link $r.target
        $bytes += $sz
    }
    Write-Host ("搬迁完成，共移动 {0:N1} GB。" -f ($bytes / 1GB)) -ForegroundColor Green
    return
}

if ($doApply) {
    Write-Host ''
    Write-Host '=== 重接（-Apply）===' -ForegroundColor Cyan
    foreach ($r in $rows) {
        $rel = $r.rel
        if ($r.state -eq 'real') {
            Write-Host ("  跳过 {0}：还是实体目录，请改用 -Migrate" -f $rel) -ForegroundColor Red
            continue
        }
        if ($r.state -eq 'junction') { Remove-Junction $r.link }
        if ($r.tstate -eq 'missing' -and -not $Force) {
            Write-Host ("  跳过 {0}：目标不存在 {1}（若确认要建空目录，追加 -Force）" -f $rel, $r.target) -ForegroundColor Yellow
            continue
        }
        New-Junction $r.link $r.target
        Write-Host ("  已联接 {0} → {1}" -f $rel, $r.target) -ForegroundColor Green
    }
    Write-Host '重接完成。' -ForegroundColor Green
    return
}
