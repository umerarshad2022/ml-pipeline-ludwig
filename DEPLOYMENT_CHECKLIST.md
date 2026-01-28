# ✅ Deployment Checklist - Linux Test Machine

## Pre-Deployment (On Windows)

### ✅ GitHub Repository Setup

- [ ] Create GitHub repository: `ludwig-ml-pipeline`
- [ ] Initialize git in project folder
- [ ] Add all files to git
- [ ] Make initial commit
- [ ] Push to GitHub
- [ ] Verify files on GitHub

**Commands**:
```powershell
cd C:\Users\umera\ml-pipeline-ludwig
git init
git add .
git commit -m "Initial commit: Ludwig ML Pipeline for AIMLFW"
git remote add origin https://github.com/YOUR_USERNAME/ludwig-ml-pipeline.git
git push -u origin main
```

---

## Deployment (On Linux Test Machine)

### ✅ System Preparation

- [ ] SSH into Linux test machine
- [ ] Verify system specs:
  ```bash
  free -h        # Should show ~16 GB RAM
  nproc          # Should show 8 CPUs
  df -h          # Should show ~100 GB disk
  python3 --version  # Should be 3.9+
  ```

### ✅ Clone Repository

- [ ] Clone from GitHub
- [ ] Navigate to directory
- [ ] Make scripts executable

**Commands**:
```bash
git clone https://github.com/YOUR_USERNAME/ludwig-ml-pipeline.git
cd ludwig-ml-pipeline
chmod +x install.sh deploy.sh
```

### ✅ Installation

**Option A: Quick Install (Recommended)**
- [ ] Run installation script
- [ ] Activate virtual environment
- [ ] Verify installation

**Commands**:
```bash
./install.sh
source venv/bin/activate
python -c "import ludwig; print(f'Ludwig: {ludwig.__version__}')"
```

**Option B: Docker Install**
- [ ] Install Docker
- [ ] Build containers
- [ ] Start services

**Commands**:
```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose
sudo systemctl start docker
docker-compose build
docker-compose up -d
```

### ✅ Initial Test

- [ ] Activate environment
- [ ] Run example pipeline
- [ ] Verify results
- [ ] Check artifacts

**Commands**:
```bash
source venv/bin/activate
python run_pipeline.py examples/du-cell-ludwig.yaml
ls -lh artifacts/ludwig_model/
```

**Expected Output**:
```
✅ Pipeline completed successfully
✅ Model saved to: artifacts/ludwig_model/
✅ Training time: 1-3 minutes
```

### ✅ Production Deployment (Optional)

- [ ] Run deployment script
- [ ] Verify service installed
- [ ] Start service
- [ ] Check status

**Commands**:
```bash
sudo ./deploy.sh
sudo systemctl status ludwig-pipeline
sudo journalctl -u ludwig-pipeline -f
```

---

## Post-Deployment Verification

### ✅ Resource Usage

- [ ] Check RAM usage: `free -h`
- [ ] Check CPU usage: `htop` or `top`
- [ ] Check disk usage: `df -h`
- [ ] Monitor processes: `ps aux | grep python`

**Expected**:
```
RAM usage:  2-4 GB during training (12 GB limit)
CPU usage:  20-60% across cores
Disk usage: < 5 GB for artifacts
```

### ✅ Functionality

- [ ] Data loading works
- [ ] Preprocessing works
- [ ] Training completes
- [ ] Evaluation works
- [ ] Artifacts saved

**Verify**:
```bash
# Check artifacts
ls -lh artifacts/ludwig_model/
cat artifacts/ludwig_model/training_statistics.json | head -20

# Check logs
tail -100 logs/pipeline.log  # If exists
```

### ✅ Performance

- [ ] Training time acceptable (1-3 min for 276 samples)
- [ ] No memory errors
- [ ] No disk space errors
- [ ] No timeout errors

**Benchmark** (276 samples):
```
Data loading:    < 1 second
Preprocessing:   < 2 seconds  
Training:        30-120 seconds
Evaluation:      < 5 seconds
Total:           ~2-3 minutes
```

---

## Troubleshooting Checks

### ❌ If Installation Fails

- [ ] Python version >= 3.9?
  ```bash
  python3 --version
  ```

- [ ] Enough disk space?
  ```bash
  df -h /
  ```

- [ ] Internet connection OK?
  ```bash
  ping -c 3 google.com
  ```

- [ ] pip working?
  ```bash
  pip --version
  ```

**Fix**:
```bash
# Update system
sudo apt-get update
sudo apt-get upgrade

# Install Python 3.9+ if needed
sudo apt-get install python3.9 python3.9-venv python3-pip
```

### ❌ If Training Fails

- [ ] Enough RAM?
  ```bash
  free -h
  ```

- [ ] Data file exists?
  ```bash
  ls -lh data/du_cell_report.csv
  ```

- [ ] Virtual environment activated?
  ```bash
  which python  # Should show venv path
  ```

**Fix**:
```bash
# Activate venv
source venv/bin/activate

# Reduce batch size in YAML
# Edit examples/du-cell-ludwig.yaml:
# trainer:
#   batch_size: 32  # Reduce if needed
```

### ❌ If Service Fails

- [ ] Service file exists?
  ```bash
  sudo systemctl status ludwig-pipeline
  ```

- [ ] Permissions OK?
  ```bash
  ls -la /opt/ludwig-ml-pipeline/
  ```

- [ ] Logs showing errors?
  ```bash
  sudo journalctl -u ludwig-pipeline -n 100
  ```

**Fix**:
```bash
# Restart service
sudo systemctl restart ludwig-pipeline

# Check logs for specific error
sudo journalctl -u ludwig-pipeline -f
```

---

## Optimization Checks

### ✅ RAM Optimization (16 GB Available)

Current settings in `examples/du-cell-ludwig.yaml`:
```yaml
trainer:
  batch_size: 32    # ✅ Can increase to 64 or 128
  num_workers: 4    # ✅ Can increase to 6
```

**Test**:
```bash
# Try larger batch size
# Edit YAML, then run:
python run_pipeline.py examples/du-cell-ludwig.yaml

# Monitor RAM
watch -n 1 'free -h'
```

### ✅ CPU Optimization (8 CPUs Available)

```bash
# Run with CPU affinity (use 6 cores)
taskset -c 0-5 python run_pipeline.py examples/du-cell-ludwig.yaml

# Monitor CPUs
htop
```

### ✅ Disk Optimization (100 GB Available)

```bash
# Check artifact sizes
du -sh artifacts/

# Clean old runs if needed
rm -rf artifacts/old_runs/

# Set up log rotation
sudo apt-get install logrotate
```

---

## Security Checks

### ✅ File Permissions

- [ ] Scripts executable: `chmod +x *.sh`
- [ ] Data protected: `chmod 700 data/`
- [ ] Configs secured: `chmod 600 examples/*.yaml`
- [ ] Logs restricted: `chmod 640 logs/*.log`

### ✅ Network Security

- [ ] Firewall configured (if needed)
- [ ] Only required ports open
- [ ] Services not exposed publicly

**Check**:
```bash
# View open ports
sudo netstat -tulpn | grep python

# Check firewall
sudo ufw status
```

---

## Monitoring Setup

### ✅ Resource Monitoring

- [ ] Install htop: `sudo apt-get install htop`
- [ ] Install sysstat: `sudo apt-get install sysstat`
- [ ] Enable monitoring: `sudo systemctl enable sysstat`

**Usage**:
```bash
# Real-time monitoring
htop

# CPU stats
mpstat 1

# Memory stats
vmstat 1

# Disk I/O
iostat -x 1
```

### ✅ Log Monitoring

- [ ] Application logs: `logs/pipeline.log`
- [ ] System logs: `/var/log/syslog`
- [ ] Service logs: `journalctl -u ludwig-pipeline`

**Setup**:
```bash
# Monitor in real-time
tail -f logs/pipeline.log

# Or with service
sudo journalctl -u ludwig-pipeline -f
```

---

## Backup Checklist

### ✅ What to Backup

- [ ] Configuration files: `examples/*.yaml`
- [ ] Training data: `data/`
- [ ] Trained models: `artifacts/ludwig_model/`
- [ ] Results: `results/`
- [ ] Logs: `logs/`

**Backup Script**:
```bash
# Create backup
tar -czf ludwig_backup_$(date +%Y%m%d).tar.gz \
  examples/ data/ artifacts/ results/ logs/

# Verify
ls -lh ludwig_backup_*.tar.gz

# Restore if needed
tar -xzf ludwig_backup_20260128.tar.gz
```

---

## Final Checklist

### ✅ Deployment Complete

- [ ] ✅ Code on GitHub
- [ ] ✅ Cloned on Linux machine
- [ ] ✅ Dependencies installed
- [ ] ✅ Test pipeline runs successfully
- [ ] ✅ Artifacts generated correctly
- [ ] ✅ Resource usage acceptable
- [ ] ✅ Performance acceptable
- [ ] ✅ Logs working
- [ ] ✅ Monitoring in place
- [ ] ✅ Backup strategy defined

### ✅ Production Ready

- [ ] ✅ Service installed (if using systemd)
- [ ] ✅ Auto-start enabled
- [ ] ✅ Security configured
- [ ] ✅ Monitoring active
- [ ] ✅ Documentation reviewed
- [ ] ✅ Team trained

---

## Quick Command Reference

```bash
# Installation
./install.sh
source venv/bin/activate

# Run pipeline
python run_pipeline.py examples/du-cell-ludwig.yaml

# Check resources
free -h          # RAM
nproc            # CPUs
df -h            # Disk
htop             # Real-time monitoring

# Service management (if deployed)
sudo systemctl start ludwig-pipeline
sudo systemctl status ludwig-pipeline
sudo journalctl -u ludwig-pipeline -f

# Docker management (if using Docker)
docker-compose up -d
docker-compose logs -f
docker-compose down

# Backup
tar -czf backup_$(date +%Y%m%d).tar.gz artifacts/ logs/

# Update from GitHub
git pull
source venv/bin/activate
pip install -r requirements.txt
```

---

## Success Criteria

Your deployment is successful if:

✅ **Installation**: Completes without errors  
✅ **Test Run**: Pipeline completes in 2-3 minutes  
✅ **Artifacts**: Model saved to `artifacts/ludwig_model/`  
✅ **Resources**: RAM < 12 GB, CPU < 80%, Disk < 10 GB  
✅ **Logs**: No errors in logs  
✅ **Repeatability**: Can run multiple times successfully  

---

## Getting Help

If you encounter issues:

1. **Check logs**: `logs/pipeline.log` or `journalctl -u ludwig-pipeline`
2. **Check resources**: `htop`, `free -h`, `df -h`
3. **Review docs**: `LINUX_DEPLOYMENT.md`, `README.md`
4. **GitHub Issues**: Open issue on your repository

---

**Your Linux test machine (16 GB RAM, 8 CPUs) is perfectly suited for Ludwig pipelines!** 🚀
