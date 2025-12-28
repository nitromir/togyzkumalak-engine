# 🚀 Togyzkumalak Engine - Deployment Guide

## Quick Start Options

| Method | Best For | Setup Time |
|--------|----------|------------|
| [Vast.ai](#vastai-deployment) | GPU Training (Recommended) | 5-10 min |
| [Docker](#docker-deployment) | Any Cloud Provider | 5 min |
| [Local](#local-development) | Development | 2 min |

---

## 🌩️ Vast.ai Deployment

### Step 1: Create Instance

1. Go to [Vast.ai](https://vast.ai/) and sign in
2. Click **"Create"** → **"New Instance"**
3. Select template: **PyTorch (Vast)**
   - Image: `vastai/pytorch`
   - This includes PyTorch with CUDA pre-installed

4. Choose your GPU:
   - **For Training**: RTX 4090 (24GB) or better
   - **For Inference Only**: RTX 3080 (10GB) is sufficient
   - Your 12x RTX 4090 instance is perfect for parallel training!

5. Configure instance:
   - **Disk Space**: 50+ GB (for models and training data)
   - **On-Start Script**: Copy contents from `vastai_onstart.sh`

### Step 2: Connect via SSH

```bash
# Vast.ai provides SSH command, something like:
ssh -p <port> root@<ip-address> -L 8000:localhost:8000
```

The `-L 8000:localhost:8000` creates a tunnel so you can access the UI locally.

### Step 3: Setup (First Time Only)

```bash
# Download and run setup script
wget https://raw.githubusercontent.com/YOUR_USERNAME/Toguzkumalak/master/gym-togyzkumalak-master/togyzkumalak-engine/deploy/vastai_setup.sh
chmod +x vastai_setup.sh
./vastai_setup.sh

# Edit .env to add your Gemini API key
nano /workspace/togyzkumalak/gym-togyzkumalak-master/togyzkumalak-engine/.env
```

### Step 4: Start Server

```bash
cd /workspace/togyzkumalak/gym-togyzkumalak-master/togyzkumalak-engine

# Option A: Direct (will stop when you disconnect)
python run.py

# Option B: Screen (persists after disconnect)
screen -S togyzkumalak
python run.py
# Press Ctrl+A, then D to detach
# Use 'screen -r togyzkumalak' to reattach

# Option C: Background with nohup
nohup python run.py > server.log 2>&1 &
```

### Step 5: Access UI

- **With SSH Tunnel**: http://localhost:8000
- **Direct IP**: http://\<vast-ip\>:8000 (ensure port 8000 is open)

---

## 🐳 Docker Deployment

### Local Docker (with GPU)

```bash
cd gym-togyzkumalak-master/togyzkumalak-engine

# Build image
docker build -t togyzkumalak .

# Run with GPU support
docker run -d \
  --name togyzkumalak \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/training_data:/app/training_data \
  -v $(pwd)/logs:/app/logs \
  -e GEMINI_API_KEY=your_key_here \
  togyzkumalak
```

### Docker Compose

```bash
# Set your Gemini API key
export GEMINI_API_KEY=your_key_here

# Start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Push to Docker Hub (for cloud deployment)

```bash
# Tag and push
docker tag togyzkumalak your_dockerhub_username/togyzkumalak:latest
docker push your_dockerhub_username/togyzkumalak:latest

# Then on any server:
docker pull your_dockerhub_username/togyzkumalak:latest
docker run -d --gpus all -p 8000:8000 your_dockerhub_username/togyzkumalak:latest
```

---

## 💻 Local Development

```bash
cd gym-togyzkumalak-master/togyzkumalak-engine

# Create virtual environment
python -m venv ../venv
..\venv\Scripts\activate  # Windows
# source ../venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run
python run.py
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
| `GEMINI_API_KEY` | - | Google Gemini API key for analysis |
| `CUDA_VISIBLE_DEVICES` | all | Which GPUs to use (e.g., `0,1,2`) |
| `LOG_LEVEL` | `info` | Logging level (debug/info/warning/error) |
| `DEV_MODE` | `false` | Enable hot-reload for development |

### .env File Example

```env
GEMINI_API_KEY=AIzaSy...your_key_here
HOST=0.0.0.0
PORT=8000
CUDA_VISIBLE_DEVICES=0,1,2,3
LOG_LEVEL=info
```

---

## 🏋️ Training on GPU

### Start AlphaZero Training via UI

1. Open http://localhost:8000
2. Go to **🧠 Тренировка** tab
3. Configure:
   - **Iterations**: 100+ (more = better)
   - **Games per iteration**: 50-100
   - **MCTS simulations**: 100-200
   - **Enable Bootstrap**: ✅ (uses human game data)
4. Click **Start Training**

### Start Training via API

```bash
curl -X POST "http://localhost:8000/api/training/alphazero/start" \
  -H "Content-Type: application/json" \
  -d '{
    "iterations": 100,
    "games_per_iteration": 100,
    "mcts_simulations": 200,
    "batch_size": 256,
    "use_bootstrap": true
  }'
```

### Monitor Training

```bash
# GPU utilization
watch -n 1 nvidia-smi

# Training metrics
curl http://localhost:8000/api/training/alphazero/metrics

# Or use the UI dashboard
```

---

## 🔧 Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size or MCTS simulations
# Or use fewer GPUs
export CUDA_VISIBLE_DEVICES=0  # Use only first GPU
```

### Port Already in Use

```bash
# Find and kill process using port 8000
# Linux/Mac:
lsof -i :8000 | awk 'NR>1 {print $2}' | xargs kill -9

# Windows:
netstat -ano | findstr :8000
taskkill /PID <pid> /F
```

### Models Not Loading

```bash
# Ensure models directory exists and has correct permissions
ls -la models/
chmod -R 755 models/
```

### Gemini API Not Working

1. Get API key from https://aistudio.google.com/
2. Set in environment: `export GEMINI_API_KEY=your_key`
3. Restart server

---

## 📊 Recommended Instance Specs

| Use Case | GPUs | RAM | Storage | Cost Estimate |
|----------|------|-----|---------|---------------|
| Quick Demo | 1x RTX 3080 | 16GB | 20GB | ~$0.20/hr |
| Training (Basic) | 1x RTX 4090 | 32GB | 50GB | ~$0.40/hr |
| Training (Fast) | 4x RTX 4090 | 128GB | 100GB | ~$1.60/hr |
| Training (Intense) | 12x RTX 4090 | 512GB | 500GB | ~$4.80/hr |

Your 12x RTX 4090 instance at $4.805/hr is excellent for intensive AlphaZero training!

---

## 🔗 Useful Links

- [Vast.ai Console](https://vast.ai/console/)
- [Google AI Studio](https://aistudio.google.com/) - Get Gemini API Key
- [PyTorch CUDA Guide](https://pytorch.org/get-started/locally/)
- [Docker GPU Support](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
