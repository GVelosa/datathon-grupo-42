# scripts/demo.ps1 - Demo completo do Datathon Grupo 42 (Windows PowerShell)
#
# Uso:
#   .\scripts\demo.ps1
#   .\scripts\demo.ps1 -SkipTrain    # pula treino se modelo ja existe
#   .\scripts\demo.ps1 -SkipTests    # pula smoke tests
#   .\scripts\demo.ps1 -SkipInstall  # pula criacao de venv e pip install
#   .\scripts\demo.ps1 -GPU          # sobe vLLM tambem (requer GPU NVIDIA)

param(
    [switch]$SkipTrain,
    [switch]$SkipTests,
    [switch]$SkipInstall,
    [switch]$GPU
)

$ErrorActionPreference = "Stop"

function Write-Step($n, $total, $msg) {
    Write-Host ""
    Write-Host "[$n/$total] $msg" -ForegroundColor Cyan
}

function Write-OK($msg)   { Write-Host "    OK: $msg"    -ForegroundColor Green  }
function Write-Warn($msg) { Write-Host "    AVISO: $msg" -ForegroundColor Yellow }
function Write-Fail($msg) { Write-Host "    ERRO: $msg"  -ForegroundColor Red    }

# Calcula total de steps dinamicamente conforme flags
$TOTAL_STEPS = 6
if (-not $SkipTrain) { $TOTAL_STEPS++ }
if (-not $SkipTests) { $TOTAL_STEPS++ }
$step = 0

# Forca UTF-8 em todos os subprocessos Python (necessario no Windows para emojis do MLflow 3.x)
$env:PYTHONUTF8 = "1"

# Garante que Docker esteja no PATH
$dockerBin = "C:\Program Files\Docker\Docker\resources\bin"
if ((Test-Path $dockerBin) -and ($env:PATH -notlike "*$dockerBin*")) {
    $env:PATH += ";$dockerBin"
}

Write-Host ""
Write-Host "=================================================" -ForegroundColor Magenta
Write-Host "  Datathon Grupo 42 - Demo Completo"             -ForegroundColor Magenta
Write-Host "  FIAP MLET Pos-Tech | Fase 5"                   -ForegroundColor Magenta
Write-Host "=================================================" -ForegroundColor Magenta

# --------------------------------------------------------------------------
# Passo: Verificar .env
# --------------------------------------------------------------------------
Write-Step (++$step) $TOTAL_STEPS ".env - verificando configuracao"

if (-not (Test-Path ".env")) {
    Copy-Item .env.example .env
    Write-Warn ".env criado de .env.example. Configure as chaves antes de continuar."
    Write-Host ""
    Write-Host "    Obrigatorio para agente com vLLM:" -ForegroundColor White
    Write-Host "      HF_TOKEN=hf_...  (token HuggingFace para baixar Llama 3.1)" -ForegroundColor Gray
    Write-Host ""
    Write-Host "    Obrigatorio para avaliacao (RAGAS/Judge):" -ForegroundColor White
    Write-Host "      ANTHROPIC_API_KEY=sk-ant-..." -ForegroundColor Gray
    Write-Host ""
    Write-Host "    Sem nenhum - sistema funciona em modo mock (sem GenAI real)" -ForegroundColor Gray
    Write-Host ""
    notepad .env
    $null = Read-Host "Pressione ENTER quando terminar de editar o .env"
} else {
    Write-OK ".env ja existe"
}

# Carrega variaveis do .env na sessao atual (ex: MLFLOW_TRACKING_URI)
Get-Content ".env" | Where-Object { $_ -match "^\s*[^#]\S+=\S" } | ForEach-Object {
    $parts = $_ -split "=", 2
    $key   = $parts[0].Trim()
    $val   = $parts[1].Trim()
    [System.Environment]::SetEnvironmentVariable($key, $val, "Process")
}
Write-Host "    Variaveis do .env carregadas na sessao" -ForegroundColor DarkGray

# --------------------------------------------------------------------------
# Passo: Verificar dataset
# --------------------------------------------------------------------------
Write-Step (++$step) $TOTAL_STEPS "Dataset - verificando creditcard.csv"

if (-not (Test-Path "data/raw/creditcard.csv")) {
    # Tenta descompactar o .gz incluido no repositorio
    if (Test-Path "data/raw/creditcard.csv.gz") {
        Write-Host "    Descompactando creditcard.csv.gz..." -ForegroundColor DarkGray
        python -c "
import gzip, shutil
with gzip.open('data/raw/creditcard.csv.gz', 'rb') as f_in, open('data/raw/creditcard.csv', 'wb') as f_out:
    shutil.copyfileobj(f_in, f_out)
"
        if ($LASTEXITCODE -ne 0) { Write-Fail "Falha ao descompactar creditcard.csv.gz"; exit 1 }
        Write-OK "creditcard.csv descompactado"
    } else {
        Write-Warn "Dataset nao encontrado em data/raw/creditcard.csv"
        Write-Host ""
        Write-Host "    Baixe em: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud" -ForegroundColor Yellow
        Write-Host "    Salve em: data\raw\creditcard.csv" -ForegroundColor Yellow
        Write-Host ""

        if (Get-Command kaggle -ErrorAction SilentlyContinue) {
            Write-Host "    Kaggle CLI detectado. Baixando automaticamente..." -ForegroundColor Cyan
            kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw/ --unzip
            if (Test-Path "data/raw/creditcard.csv") {
                Write-OK "Dataset baixado via Kaggle CLI"
            }
        } else {
            Write-Host "    Para download automatico: pip install kaggle" -ForegroundColor Gray
            Write-Host "    Depois: kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw/ --unzip" -ForegroundColor Gray
            Write-Host ""
            $null = Read-Host "Pressione ENTER apos salvar o arquivo"
            if (-not (Test-Path "data/raw/creditcard.csv")) {
                Write-Fail "Dataset ainda nao encontrado. Abortando."
                exit 1
            }
        }
    }
} else {
    $sizeMB = [math]::Round((Get-Item "data/raw/creditcard.csv").Length / 1MB, 1)
    Write-OK "creditcard.csv encontrado ($sizeMB MB)"
}

# --------------------------------------------------------------------------
# Passo: Venv + dependencias
# --------------------------------------------------------------------------
if (-not $SkipInstall) {
    Write-Step (++$step) $TOTAL_STEPS "Python - criando venv e instalando dependencias"

    if (-not (Test-Path ".venv")) {
        Write-Host "    Criando .venv..." -ForegroundColor DarkGray
        python -m venv .venv
        if ($LASTEXITCODE -ne 0) { Write-Fail "python -m venv falhou"; exit 1 }
    } else {
        Write-Host "    .venv ja existe, reusando..." -ForegroundColor DarkGray
    }

    . .\.venv\Scripts\Activate.ps1
    Write-Host "    venv ativada: $env:VIRTUAL_ENV" -ForegroundColor DarkGray

    pip install -e ".[dev,eval,monitoring]" -q
    if ($LASTEXITCODE -ne 0) { Write-Fail "pip install falhou"; exit 1 }
    Write-OK "Dependencias instaladas na venv"
} else {
    Write-Step (++$step) $TOTAL_STEPS "Python - pulando instalacao (-SkipInstall)"

    if (Test-Path ".venv\Scripts\Activate.ps1") {
        . .\.venv\Scripts\Activate.ps1
        Write-OK "venv ativada: $env:VIRTUAL_ENV"
    } else {
        Write-Warn "Nenhuma .venv encontrada - usando Python global"
    }
}

# --------------------------------------------------------------------------
# Passo: Subir Docker
# --------------------------------------------------------------------------
Write-Step (++$step) $TOTAL_STEPS "Docker - subindo servicos (MLflow + Prometheus + Grafana)"

if ($GPU) {
    Write-Host "    Modo GPU: subindo vLLM tambem..." -ForegroundColor Yellow
    docker compose --profile gpu up -d
} else {
    docker compose up -d mlflow prometheus grafana
}
if ($LASTEXITCODE -ne 0) { Write-Fail "docker compose up falhou. Docker Desktop esta rodando?"; exit 1 }
Write-OK "Containers iniciados"

# --------------------------------------------------------------------------
# Passo: Aguardar MLflow
# --------------------------------------------------------------------------
Write-Step (++$step) $TOTAL_STEPS "MLflow - aguardando disponibilidade (max. 60s)"

$maxWait  = 60
$waited   = 0
$mlflowOk = $false
while ($waited -lt $maxWait) {
    Start-Sleep -Seconds 3
    $waited += 3
    try {
        $null = Invoke-RestMethod -Uri "http://localhost:5000/api/2.0/mlflow/experiments/search" -Method POST -ContentType "application/json" -Body '{"max_results":1}' -TimeoutSec 2 -ErrorAction Stop
        $mlflowOk = $true
        break
    } catch { }
    Write-Host "    aguardando... ${waited}s" -ForegroundColor DarkGray
}

if (-not $mlflowOk) {
    Write-Warn "MLflow nao respondeu em ${maxWait}s. Continuando de qualquer forma..."
} else {
    Write-OK "MLflow disponivel em http://localhost:5000"
}

# --------------------------------------------------------------------------
# Passo: Treinar modelo
# --------------------------------------------------------------------------
if (-not $SkipTrain) {
    Write-Step (++$step) $TOTAL_STEPS "Treino - RF + MLP (loga no MLflow do Docker)"
    Write-Host "    Isso pode levar 2-5 minutos..." -ForegroundColor DarkGray
    Write-Host "    MLFLOW_TRACKING_URI = $env:MLFLOW_TRACKING_URI" -ForegroundColor DarkGray

    python scripts/train.py
    if ($LASTEXITCODE -ne 0) { Write-Fail "Treino falhou"; exit 1 }
    Write-OK "Modelo treinado e registrado no MLflow"
}

# --------------------------------------------------------------------------
# Passo: Subir fraud-api
# --------------------------------------------------------------------------
Write-Step (++$step) $TOTAL_STEPS "fraud-api - subindo container"

docker compose up -d fraud-api
if ($LASTEXITCODE -ne 0) { Write-Fail "docker compose up fraud-api falhou"; exit 1 }

Write-Host "    Aguardando API ficar disponivel (max. 60s)..." -ForegroundColor DarkGray
$waited = 0
$apiOk  = $false
while ($waited -lt 60) {
    Start-Sleep -Seconds 3
    $waited += 3
    try {
        $resp = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 2 -ErrorAction Stop
        if ($resp.status -eq "healthy") { $apiOk = $true; break }
    } catch { }
    Write-Host "    aguardando... ${waited}s" -ForegroundColor DarkGray
}

if (-not $apiOk) {
    Write-Warn "API nao respondeu em 60s. Verifique: docker compose logs fraud-api"
} else {
    Write-OK "API disponivel em http://localhost:8000"
}

# --------------------------------------------------------------------------
# Passo: Smoke tests
# --------------------------------------------------------------------------
if (-not $SkipTests) {
    Write-Step (++$step) $TOTAL_STEPS "Smoke tests - validando endpoints"
    python scripts/smoke_api.py
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "Alguns smoke tests falharam. Verifique os logs acima."
    } else {
        Write-OK "Todos os smoke tests passaram"
    }
}

# --------------------------------------------------------------------------
# Resumo
# --------------------------------------------------------------------------
Write-Host ""
Write-Host "=================================================" -ForegroundColor Green
Write-Host "  Stack pronta! Acesse:"                          -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Green
Write-Host "  API (Swagger UI):  http://localhost:8000/docs" -ForegroundColor White
Write-Host "  MLflow:            http://localhost:5000"       -ForegroundColor White
Write-Host "  Grafana:           http://localhost:3000   (admin / datathon42)" -ForegroundColor White
Write-Host "  Prometheus:        http://localhost:9090"       -ForegroundColor White
if ($GPU) {
    Write-Host "  vLLM (OpenAI API): http://localhost:8080/v1" -ForegroundColor White
}
Write-Host ""
Write-Host "  Para parar tudo: docker compose down" -ForegroundColor DarkGray
Write-Host ""
