
# install

```bash
conda env create -f env.yaml
conda activate pet-baseline
git clone https://github.com/huggingface/neuralcoref.git
cd neuralcoref
pip install -e .
```

# running
```bash
conda activate pet-baseline
python main.py
```