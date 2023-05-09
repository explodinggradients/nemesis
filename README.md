# Reward-Model
Reward Model training framework for LLM RLHF. For in-depth understanding of Reward modeling, checkout our [blog](https://explodinggradients.com/)
The word nemesis originally meant the distributor of fortune, neither good nor bad, simply in due proportion to each according to what was deserved.
### Quick Start
* Inference
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
MODEL = "shahules786/Reward-model-gptneox-410M"

model = AutoModelForSequenceClassification.from_pretrained(MODEL)
tokenizer = AutoTokenizer.from_pretrained(MODEL)

```

* Training
```bash
python src/training.py --config-name <your-config-name>
```



## Contributions
* All contributions are welcome. Checkout #issues
