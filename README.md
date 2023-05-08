# Reward-Model
Framework for reward model for RLHF. 


### Quick Start
* Inference
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
MODEL = ""

model = AutoModelForSequenceClassification.from_pretrained(MODEL)
tokenizer = AutoTokenizer.from_pretrained(MODEL)

```

* Training
```bash
python src/training.py --config-name <your-config-name>
```

