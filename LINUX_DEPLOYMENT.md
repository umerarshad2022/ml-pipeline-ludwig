# 🐧 Linux Deployment Guide

## System Requirements

**Your Test Machine** ✅
```
RAM:  16 GB  ✓
HDD:  100 GB ✓
CPUs: 8      ✓
OS:   Linux (Unix-based)
```

**Minimum Requirements**:
- Python 3.9+
- 8 GB RAM (you have 16 GB ✓)
- 20 GB free disk space (you have 100 GB ✓)
- 4 CPUs (you have 8 ✓)

---

## Quick Start (3 Commands)

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/ludwig-ml-pipeline.git
cd ludwig-ml-pipeline

# 2. Run installation script
chmod +x install.sh
./install.sh

# 3. Run pipeline
source venv/bin/activate
python run_pipeline.py examples/du-cell-ludwig.yaml
```

**That's it!** ✅

---

## Detailed Installation

### Option 1: Automated Installation (Recommended)

```bash
# Make script executable
chmod +x install.sh

# Run installation
./install.sh

# Script will:
# - Check Python version
# - Create virtual environment
# - Install PyTorch (CPU version)
# - Install Ludwig and dependencies
# - Create necessary directories
# - Validate installation
```

### Option 2: Manual Installation

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Upgrade pip
pip install --upgrade pip setuptools wheel

# 3. Install PyTorch (CPU version for your system)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. Install Ludwig
pip install ludwig[serve]==0.8.5

# 5. Install other dependencies
pip install -r requirements.txt

# 6. Create directories
mkdir -p artifacts logs data results

# 7. Test installation
python -c "import ludwig; print(f'Ludwig version: {ludwig.__version__}')"
```

### Option 3: Docker Installation

```bash
# 1. Install Docker (if not already installed)
sudo apt-get update
sudo apt-get install -y docker.io docker-compose

# 2. Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

# 3. Build and run
docker-compose up -d

# 4. Check logs
docker-compose logs -f ludwig-pipeline
```

---

## Resource Optimization (16 GB RAM, 8 CPUs)

### Ludwig Configuration Tuning

Edit `examples/du-cell-ludwig.yaml`:

```yaml
trainer:
  # Optimize for your 16 GB RAM
  batch_size: 64          # ← Larger batches (you have RAM!)
  epochs: 100
  
  # Utilize your 8 CPUs
  num_workers: 6          # ← Leave 2 CPUs for system
  
  # Training optimizations
  learning_rate: 0.001
  early_stop: 10
```

### System-Level Optimizations

```bash
# 1. Set CPU affinity (use 6 cores for Ludwig)
taskset -c 0-5 python run_pipeline.py examples/du-cell-ludwig.yaml

# 2. Monitor resource usage
htop  # or 'top'

# 3. Set memory limits
ulimit -v 14680064  # Limit to 14 GB (leave 2 GB for system)
```

### Docker Resource Limits

Already configured in `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '6'        # Use 6 of your 8 CPUs
      memory: 12G      # Use 12 of your 16 GB
    reservations:
      cpus: '2'
      memory: 4G
```

---

## Running the Pipeline

### Basic Usage

```bash
# Activate virtual environment
source venv/bin/activate

# Run pipeline
python run_pipeline.py examples/du-cell-ludwig.yaml

# View results
ls -lh artifacts/ludwig_model/
```

### Advanced Usage

```bash
# Run with custom config
python run_pipeline.py path/to/your/config.yaml

# Run in background
nohup python run_pipeline.py examples/du-cell-ludwig.yaml > logs/pipeline.log 2>&1 &

# Monitor progress
tail -f logs/pipeline.log
```

### Docker Usage

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f ludwig-pipeline

# Stop services
docker-compose down

# Remove all data
docker-compose down -v
```

---

## Production Deployment

### Deploy as Systemd Service

```bash
# Make deployment script executable
chmod +x deploy.sh

# Run deployment (needs sudo)
sudo ./deploy.sh

# Service will be installed at: /opt/ludwig-ml-pipeline
```

**Service Commands**:
```bash
# Start service
sudo systemctl start ludwig-pipeline

# Stop service
sudo systemctl stop ludwig-pipeline

# Check status
sudo systemctl status ludwig-pipeline

# View logs
sudo journalctl -u ludwig-pipeline -f

# Enable auto-start on boot
sudo systemctl enable ludwig-pipeline
```

### Manual Service Setup

1. Create service file:
```bash
sudo nano /etc/systemd/system/ludwig-pipeline.service
```

2. Add content:
```ini
[Unit]
Description=Ludwig ML Pipeline Service
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/ludwig-ml-pipeline
Environment="PATH=/path/to/ludwig-ml-pipeline/venv/bin"
ExecStart=/path/to/ludwig-ml-pipeline/venv/bin/python run_pipeline.py examples/du-cell-ludwig.yaml
Restart=on-failure
RestartSec=10
MemoryLimit=12G
CPUQuota=600%

[Install]
WantedBy=multi-user.target
```

3. Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable ludwig-pipeline
sudo systemctl start ludwig-pipeline
```

---

## Monitoring & Logging

### Log Locations

```bash
# Application logs
logs/pipeline.log

# Ludwig training logs
artifacts/ludwig_model/training_log.json

# Docker logs
docker-compose logs -f

# Systemd logs
sudo journalctl -u ludwig-pipeline -f
```

### Resource Monitoring

```bash
# Real-time monitoring
htop

# CPU usage
mpstat 1

# Memory usage
free -h
watch -n 1 'free -h'

# Disk usage
df -h
du -sh artifacts/

# Process monitoring
ps aux | grep python

# Network monitoring (if using Docker)
docker stats
```

---

## Troubleshooting

### Issue: Out of Memory

```bash
# Reduce batch size in YAML
trainer:
  batch_size: 32  # ← Reduce from 64

# Or use Docker with memory limit
docker-compose up  # Already limited to 12G
```

### Issue: Slow Training

```bash
# Increase workers to use more CPUs
trainer:
  num_workers: 7  # ← Use more of your 8 CPUs
  batch_size: 128  # ← Larger batches (you have RAM!)
```

### Issue: Disk Space

```bash
# Check disk usage
df -h
du -sh artifacts/ logs/

# Clean old artifacts
rm -rf artifacts/old_runs/

# Clean Docker volumes
docker system prune -a --volumes
```

### Issue: Permission Denied

```bash
# Make scripts executable
chmod +x install.sh deploy.sh

# Fix file permissions
chmod -R 755 pipelines/
chmod 644 requirements.txt

# Fix data directory
chmod -R 755 data/
```

### Issue: Python Version

```bash
# Check Python version
python3 --version

# If too old, install newer version
sudo apt-get update
sudo apt-get install -y python3.9 python3.9-venv

# Use specific version
python3.9 -m venv venv
```

---

## Performance Benchmarks

### Expected Performance (16 GB RAM, 8 CPUs)

```
Data Loading:      < 1 second (276 samples)
Preprocessing:     < 2 seconds
Training:          30 seconds - 2 minutes
Evaluation:        < 5 seconds
Total:             ~2-3 minutes
```

### Scaling Estimates

| Samples | RAM Usage | Training Time | CPUs Used |
|---------|-----------|---------------|-----------|
| 276     | ~2 GB     | 30-60 sec     | 2-4       |
| 1,000   | ~3 GB     | 1-2 min       | 4-6       |
| 10,000  | ~6 GB     | 5-10 min      | 6-8       |
| 100,000 | ~12 GB    | 30-60 min     | 8         |

**Your system can handle up to ~100K samples comfortably!**

---

## Security Best Practices

### File Permissions

```bash
# Secure configuration files
chmod 600 examples/*.yaml

# Secure data directory
chmod 700 data/

# Secure logs
chmod 600 logs/*.log
```

### Environment Variables

```bash
# Create .env file for secrets
cat > .env <<EOF
LUDWIG_CACHE_DIR=/opt/ludwig_cache
MLFLOW_TRACKING_URI=http://localhost:5000
AWS_ACCESS_KEY_ID=your_minio_key
AWS_SECRET_ACCESS_KEY=your_minio_secret
EOF

# Secure .env
chmod 600 .env

# Load in scripts
source .env
```

### Docker Security

```bash
# Run as non-root user
docker-compose exec ludwig-pipeline whoami

# Scan for vulnerabilities
docker scan ludwig-ml-pipeline:latest
```

---

## Backup & Recovery

### Backup Artifacts

```bash
# Manual backup
tar -czf ludwig_backup_$(date +%Y%m%d).tar.gz artifacts/ logs/ data/

# Automated backup (cron)
# Add to crontab: crontab -e
0 2 * * * cd /path/to/ludwig-ml-pipeline && tar -czf backup_$(date +\%Y\%m\%d).tar.gz artifacts/
```

### Restore from Backup

```bash
# Extract backup
tar -xzf ludwig_backup_20260128.tar.gz

# Verify
ls -lh artifacts/ logs/ data/
```

---

## Maintenance

### Update Ludwig

```bash
# Activate environment
source venv/bin/activate

# Update Ludwig
pip install --upgrade ludwig[serve]

# Test
python -c "import ludwig; print(ludwig.__version__)"
```

### Cleanup

```bash
# Remove old artifacts (keep last 7 days)
find artifacts/ -type f -mtime +7 -delete

# Clean pip cache
pip cache purge

# Clean Docker
docker system prune -a
```

---

## Next Steps

1. **Initial Test**:
   ```bash
   ./install.sh
   source venv/bin/activate
   python run_pipeline.py examples/du-cell-ludwig.yaml
   ```

2. **Production Deployment**:
   ```bash
   sudo ./deploy.sh
   sudo systemctl start ludwig-pipeline
   ```

3. **Monitor & Optimize**:
   ```bash
   sudo journalctl -u ludwig-pipeline -f
   htop
   ```

---

##Additional Resources

- **Ludwig Docs**: https://ludwig.ai/latest/
- **Docker Docs**: https://docs.docker.com/
- **Systemd Guide**: `man systemd.service`
- **Linux Performance**: `man htop`, `man mpstat`

---

**Your 16 GB RAM, 8 CPU Linux system is perfect for Ludwig pipelines!** 🚀
