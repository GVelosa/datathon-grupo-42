# scripts/demo.ps1 — Demo completo do Datathon Grupo 42 (Windows PowerShell)
#
# Executa TUDO de uma vez:
#   1. Verifica .env e dataset
#   2. Instala dependências Python
#   3. Sobe Docker (MLflow + Prometheus + Grafana)
#   4. Aguarda MLflow ficar disponível
#   5. Treina modelo (RF + MLP, loga no MLflow do Docker)
#   6. Sobe fraud-api no Docker
#   7. Aguarda API ficar disponível
#   8. Executa smoke tests
#   9. Imprime URLs de acesso
#
# Uso:
#   .\scripts\demo.ps1
#   .\scripts\demo.ps1 -SkipTrain    # pula treino se modelo já existe
#   .\scripts\demo.ps1 -SkipTests    # pula smoke tests
#   .\scripts\demo.ps1 -GPU          # sobe vLLM também (requer GPU NVIDIA)

param(
    [switch]$SkipTrain,
    [switch]$SkipTests,
    [switch]$GPU
)

$ErrorActionPreference = "Stop"

function Write-Step($n, $total, $msg) {
    Write-Host ""
    Write-Host "[$n/$total] $msg" -ForegroundColor Cyan
}

function Write-OK($msg) { Write-Host "    OK: $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "    AVISO: $msg" -ForegroundColor Yellow }
function Write-Fail($msg) { Write-Host "    ERRO: $msg" -ForegroundColor Red }

$TOTAL_STEPS = if ($SkipTrain) { 7 } else { 8 }

Write-Host ""
Write-Host "=================================================" -ForegroundColor Magenta
Write-Host "  Datathon Grupo 42 — Demo Completo" -ForegroundColor Magenta
Write-Host "  FIAP MLET Pós-Tech | Fase 5" -ForegroundColor Magenta
Write-Host "=================================================" -ForegroundColor Magenta

# ── Passo 1: Verificar .env ────────────────────────────────────────────────
Write-Step 1 $TOTAL_STEPS ".env — verificando configuração"

if (-not (Test-Path ".env")) {
    Copy-Item .env.example .env
    Write-Warn ".env criado de .env.example. Configure as chaves antes de continuar:"
    Write-Host ""
    Write-Host "    Obrigatório para agente com vLLM:" -ForegroundColor White
    Write-Host "      HF_TOKEN=hf_...  (token HuggingFace para baixar Llama 3.1)" -ForegroundColor Gray
    Write-Host ""
    Write-Host "    Obrigatório para avaliação (RAGAS/Judge):" -ForegroundColor White
    Write-Host "      ANTHROPIC_API_KEY=sk-ant-..." -ForegroundColor Gray
    Write-Host ""
    Write-Host "    Se não tiver nenhum — sistema funciona em modo mock (sem GenAI real)" -ForegroundColor Gray
    Write-Host ""
    notepad .env
    $null = Read-Host "Pressione ENTER quando terminar de editar o .env"
} else {
    Write-OK ".env já existe"
}

# ── Passo 2: Verificar dataset ─────────────────────────────────────────────
Write-Step 2 $TOTAL_STEPS "Dataset — verificando creditcard.csv"

if (-not (Test-Path "data/raw/creditcard.csv")) {
    Write-Warn "Dataset não encontrado em data/raw/creditcard.csv"
    Write-Host ""
    Write-Host "    Baixe em: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud" -ForegroundColor Yellow
    Write-Host "    Salve o arquivo em: $(Resolve-Path .)\data\raw\creditcard.csv" -ForegroundColor Yellow
    Write-Host ""

    # Tenta Kaggle CLI se disponível
    if (Get-Command kaggle -ErrorAction SilentlyContinue) {
        Write-Host "    Kaggle CLI detectado. Baixando automaticamente..." -ForegroundColor Cyan
        kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw/ --unzip
        if (Test-Path "data/raw/creditcard.csv") {
            Write-OK "Dataset baixado via Kaggle CLI"
        }
    } else {
        Write-Host "    Instale o Kaggle CLI para download automático:" -ForegroundColor Gray
        Write-Host "    pip install kaggle && kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw/ --unzip" -ForegroundColor Gray
        Write-Host ""
        $null = Read-Host "Pressione ENTER após salvar o arquivo e continuar"
        if (-not (Test-Path "data/raw/creditcard.csv")) {
            Write-Fail "Dataset ainda não encontrado. Abortando."
            exit 1
        }
    }
} else {
    $size = [math]::Round((Get-Item "data/raw/creditcard.csv").Length / 1MB, 1)
    Write-OK "creditcard.csv encontrado (${size} MB)"
}

# ── Passo 3: Instalar dependências ─────────────────────────────────────────
Write-Step 3 $TOTAL_STEPS "Python — instalando dependências"

pip install -e ".[dev,eval,monitoring]" -q
if ($LASTEXITCODE -ne 0) { Write-Fail "pip install falhou"; exit 1 }
Write-OK "Dependências instaladas"

# ── Passo 4: Subir Docker ──────────────────────────────────────────────────
Write-Step 4 $TOTAL_STEPS "Docker — subindo serviços (MLflow + Prometheus + Grafana)"

if ($GPU) {
    Write-Host "    Modo GPU: subindo vLLM também..." -ForegroundColor Yellow
    docker compose --profile gpu up -d
} else {
    docker compose up -d mlflow prometheus grafana
}
if ($LASTEXITCODE -ne 0) { Write-Fail "docker compose up falhou. Docker Desktop está rodando?"; exit 1 }
Write-OK "Containers iniciados"

# ── Passo 5: Aguardar MLflow ───────────────────────────────────────────────
Write-Step 5 $TOTAL_STEPS "MLflow — aguardando disponibilidade (máx. 60s)"

$maxWait = 60
$waited = 0
$mlflowOk = $false
while ($waited -lt $maxWait) {
    Start-Sleep -Seconds 3
    $waited += 3
    try {
        $null = Invoke-RestMethod -Uri "http://localhost:5000/api/2.0/mlflow/experiments/list" -TimeoutSec 2 -ErrorAction Stop
        $mlflowOk = $true
        break
    } catch { }
    Write-Host "    aguardando... ${waited}s" -ForegroundColor DarkGray
}

if (-not $mlflowOk) {
    Write-Warn "MLflow não respondeu em ${maxWait}s. Continuando de qualquer forma..."
} else {
    Write-OK "MLflow disponível em http://localhost:5000"
}

# ── Passo 6: Treinar modelo ────────────────────────────────────────────────
if (-not $SkipTrain) {
    Write-Step 6 $TOTAL_STEPS "Treino — RF + MLP (loga no MLflow do Docker)"
    Write-Host "    Isso pode levar 2-5 minutos..." -ForegroundColor DarkGray

    python scripts/train.py
    if ($LASTEXITCODE -ne 0) { Write-Fail "Treino falhou"; exit 1 }
    Write-OK "Modelo treinado e registrado no MLflow"
}

# ── Passo 7: Subir fraud-api ───────────────────────────────────────────────
Write-Step 7 $TOTAL_STEPS "fraud-api — subindo container"

docker compose up -d fraud-api
if ($LASTEXITCODE -ne 0) { Write-Fail "docker compose up fraud-api falhou"; exit 1 }

Write-Host "    Aguardando API ficar disponível (máx. 60s)..." -ForegroundColor DarkGray
$waited = 0
$apiOk = $false
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
    Write-Warn "API não respondeu em 60s. Verifique: docker compose logs fraud-api"
} else {
    Write-OK "API disponível em http://localhost:8000"
}

# ── Passo 8: Smoke tests ───────────────────────────────────────────────────
if (-not $SkipTests) {
    Write-Step 8 $TOTAL_STEPS "Smoke tests — validando endpoints"
    python scripts/smoke_api.py
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "Alguns smoke tests falharam. Verifique os logs acima."
    } else {
        Write-OK "Todos os smoke tests passaram"
    }
}

# ── Resumo ─────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "=================================================" -ForegroundColor Green
Write-Host "  Stack pronta! Acesse:" -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Green
Write-Host "  API (Swagger UI):  http://localhost:8000/docs" -ForegroundColor White
Write-Host "  MLflow:            http://localhost:5000" -ForegroundColor White
Write-Host "  Grafana:           http://localhost:3000   (admin / datathon42)" -ForegroundColor White
Write-Host "  Prometheus:        http://localhost:9090" -ForegroundColor White
if ($GPU) {
    Write-Host "  vLLM (OpenAI API): http://localhost:8080/v1" -ForegroundColor White
}
Write-Host ""
Write-Host "  Para parar tudo:" -ForegroundColor DarkGray
Write-Host "  docker compose down" -ForegroundColor DarkGray
Write-Host ""
