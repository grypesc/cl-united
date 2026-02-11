# Ensemble of experts repository (aka united we stand)

This work is based on work https://arxiv.org/html/2401.10191  

Here we train a single expert but utilize knowledge distillation from all the experts into the trained one.  

# Setup  

Create virtual environment and install dependencies
```bash
python3 -m venv venv && source venv/bin/activate
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
pip install requirements.txt
```

Reproduce experiments using scripts in ```scripts``` directory:
```bash
bash scripts/cifar-10x10.sh
```
