# 🚀 Quick Start Guide - Ludwig ML Pipeline

## Welcome!

This is a **YAML-based ML pipeline engine powered by Ludwig AI** - making deep learning as easy as writing configuration files!

---

## ⚡ 5-Minute Quick Start

### Step 1: Install Dependencies
```powershell
# Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\activate

# Install Ludwig and dependencies
pip install -r requirements.txt
```

**Note**: Ludwig installation may take 5-10 minutes (it's a big package!)

### Step 2: Run Example Pipeline
```powershell
# Run DU cell anomaly detection with Ludwig
python run_pipeline.py examples\du-cell-ludwig.yaml
```

### Step 3: View Results
```powershell
# Check artifacts directory
dir artifacts\ludwig_model\

# View training statistics
type artifacts\ludwig_model\training_statistics.json
```

---

## 📊 What the Example Does

**Pipeline**: `examples/du-cell-ludwig.yaml`

```
1. Load Data         → Load du_cell_report.csv (276 samples)
2. Auto-Label        → Use Isolation Forest to create labels
3. Train Ludwig      → Train deep neural network
4. Evaluate          → Test model performance
5. Visualize         → Create plots (learning curves, etc.)
```

**Output**:
- ✅ Trained deep learning model
- ✅ Training statistics & metrics
- ✅ Evaluation results
- ✅ Visualizations (learning curves, confusion matrix)

---

## 🎯 Comparison: Scikit-learn vs Ludwig

| Feature | Original Engine (Scikit-learn) | **This Engine (Ludwig)** |
|---------|-------------------------------|-------------------------|
| **Approach** | Traditional ML | Deep Learning |
| **Configuration** | YAML | YAML (same!) |
| **Best For** | Small data (< 1000 samples) | Large data (1000+ samples) |
| **Training Time** | Fast (< 1 sec) | Slower (30 sec - 5 min) |
| **Model Complexity** | Simple (Isolation Forest) | Complex (Neural Networks) |
| **GPU Support** | No | Yes |
| **Feature Engineering** | Manual | Automatic |
| **Hyperparameter Tuning** | Manual | Automatic |

---

## 📁 Project Structure

```
ml-pipeline-ludwig/
├── pipelines/
│   ├── engine.py                  # Main Ludwig pipeline engine
│   └── handlers/
│       ├── ludwig_handler.py      # Ludwig training/evaluation
│       ├── data_loader.py         # Data loading
│       └── ludwig_visualize.py    # Visualization generation
├── examples/
│   └── du-cell-ludwig.yaml        # DU cell example (276 samples)
├── data/
│   └── du_cell_report.csv         # Your DU cell data ✅
├── artifacts/                     # Generated outputs (created on run)
├── run_pipeline.py                # CLI runner
├── requirements.txt               # Dependencies
└── README.md                      # Full documentation
```

---

## 🔧 How It Works

### Traditional Approach (Lots of Code)
```python
import tensorflow as tf
import pandas as pd

# Load data
df = pd.read_csv('data.csv')

# Preprocess
# ... 50 lines of preprocessing code ...

# Build model
model = tf.keras.Sequential([
    # ... 30 lines of model architecture ...
])

# Train
model.compile(...)
model.fit(...)

# Evaluate
# ... 20 lines of evaluation code ...
```

### Ludwig Approach (Just YAML!)
```yaml
# Load data (automatic)
data:
  source:
    type: csv
    path: data.csv

# Define model (automatic architecture)
ludwig:
  input_features:
    - name: Active_UEs_DL
      type: numerical
  output_features:
    - name: is_anomaly
      type: binary
  trainer:
    epochs: 100

# Everything else is automatic!
```

---

## 💡 Key Concepts

### 1. **Ludwig Configuration**
Ludwig uses declarative YAML to define:
- **Input features**: What data to train on
- **Output features**: What to predict
- **Trainer config**: How to train

**Ludwig automatically handles**:
- Model architecture selection
- Feature preprocessing & normalization
- Training loop & validation
- Checkpointing & logging

### 2. **Pipeline Steps**
Your YAML defines a sequence of steps:
- `data_loader`: Load CSV/Parquet/JSON
- `unsupervised_labeling`: Create labels (for unsupervised data)
- `ludwig_train`: Train deep learning model
- `ludwig_evaluate`: Evaluate performance
- `ludwig_visualize`: Create plots

### 3. **Context Passing**
Data flows between steps automatically:
```
data_loader → context['data']
  ↓
unsupervised_labeling → updates context['data'] with labels
  ↓
ludwig_train → uses context['data'], saves model
  ↓
ludwig_evaluate → uses saved model
```

---

## 📝 Your DU Cell Data

**Dataset**: `data/du_cell_report.csv`
```
Rows: 276 samples
Columns: 12 (Cell_ID, gNB_ID, Active_UEs_DL, etc.)
```

**Challenge**: No labels for anomaly detection!

**Solution**: Pipeline automatically:
1. Uses Isolation Forest to create pseudo-labels
2. Trains Ludwig neural network on these labels
3. Results in a deep learning model for anomaly detection

---

## 🎓 Learning Path

### Beginner: Run Examples
```powershell
# 1. Run basic example
python run_pipeline.py examples\du-cell-ludwig.yaml

# 2. Check results
dir artifacts\ludwig_model\

# 3. Understand what happened
type artifacts\ludwig_model\training_statistics.json
```

### Intermediate: Customize YAML
```yaml
# Edit examples/du-cell-ludwig.yaml

# Try different architectures
output_features:
  - name: is_anomaly
    decoder:
      num_fc_layers: 5  # ← Change from 3 to 5
      fc_layers:
        - output_size: 128  # ← Bigger network
        - output_size: 64
        - output_size: 32

# Try different training params
trainer:
  epochs: 200  # ← More epochs
  batch_size: 64  # ← Larger batches
  learning_rate: 0.0001  # ← Smaller learning rate
```

### Advanced: Add New Steps
```yaml
# Add your own preprocessing
steps:
  - name: custom-preprocessing
    type: custom_handler
    config:
      # Your custom logic
```

---

## 🔍 Troubleshooting

### Issue: Ludwig installation fails
```powershell
# Try installing in steps
pip install torch  # Install PyTorch first
pip install ludwig  # Then Ludwig
```

### Issue: Out of memory
```yaml
# Reduce batch size in trainer config
trainer:
  batch_size: 16  # ← Smaller batches use less RAM
```

### Issue: Training takes too long
```yaml
# Reduce epochs
trainer:
  epochs: 50  # ← Fewer epochs = faster training
```

### Issue: Model not learning
```yaml
# Try different learning rate
trainer:
  learning_rate: 0.01  # ← Higher for faster learning
  # or
  learning_rate: 0.0001  # ← Lower for more stable learning
```

---

## 🆚 When to Use Which Engine?

### Use **Original Engine** (Scikit-learn) if:
✅ Small dataset (< 1000 samples) - **Like your 276 samples!**  
✅ Need fast results (seconds)  
✅ Simple patterns  
✅ Limited compute (no GPU)  
✅ High interpretability needed  

### Use **This Engine** (Ludwig) if:
✅ Large dataset (1000+ samples)  
✅ Complex patterns  
✅ GPU available  
✅ Need state-of-the-art performance  
✅ Multi-modal data (text + numbers + images)  

### For Your 276 Samples:
**Recommendation**: Start with original engine (Isolation Forest)
- ✅ Faster
- ✅ Works great with small data
- ✅ More interpretable

**Try Ludwig** when:
- You collect more data (aim for 1000+ samples)
- Patterns become more complex
- You want to compare deep learning vs traditional ML

---

## 📚 Next Steps

### 1. **Understand Ludwig**
- Read: https://ludwig.ai/latest/
- Tutorial: https://ludwig.ai/latest/getting_started/

### 2. **Customize for Your Data**
```yaml
# Add more features when available
input_features:
  - name: Active_UEs_DL
    type: numerical
  - name: PRB_Used_DL  # ← Add when you have this
    type: numerical
  - name: Throughput_DL  # ← Add when you have this
    type: numerical
```

### 3. **Compare Results**
```powershell
# Run original engine
cd ..\ml-pipeline-engine
python test_direct_isolation_forest.py

# Run Ludwig engine
cd ..\ml-pipeline-ludwig
python run_pipeline.py examples\du-cell-ludwig.yaml

# Compare results!
```

---

## 🎯 Summary

**You now have TWO ML pipeline engines:**

| Engine | Location | Best For |
|--------|----------|----------|
| **Original** | `ml-pipeline-engine/` | Small data, fast iteration |
| **Ludwig** | `ml-pipeline-ludwig/` | Large data, deep learning |

**Both use YAML configuration! Both are production-ready!**

**Next**: Try running the Ludwig pipeline and see how it compares! 🚀
