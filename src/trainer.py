from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from transformers import Trainer
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from datacollator import RMDataCollator
import os

from loss import RMLoss
from model import GPTNeoXRM
from utils import get_tokenizer, prepare_datasets


class RMTrainer(Trainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss = RMLoss(reduction="mean")

    def compute_loss(self, model, inputs, return_outputs=False):

        k_lens = inputs.pop("k_lens")
        inputs = self._prepare_inputs(inputs)
        logits = model(**inputs).logits
        loss = self.loss(logits, k_lens)
        return (loss,logits) if return_outputs else loss
    
    def prediction_step(self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None):

        with torch.no_grad():
            loss,logits = self.compute_loss(model, inputs, return_outputs=True)
        
        return (loss,logits,None)
    

@hydra.main(version_base=None, config_path="config", config_name="config")
def train(cfg: DictConfig) -> None:
    if not os.path.exists(cfg.log_dir):
        os.mkdir(cfg.log_dir)

    if not cfg.log_wandb:
        os.environ["WANDB_MODE"] = "offline"

    if cfg.log_wandb:
        import wandb

        wandb.init(
            project="Reward-model",
            entity=cfg.wandb_entity,
            name=f"{cfg.model}-{cfg.run_name}-rm",
            config=cfg,
        )

    model = GPTNeoXRM.from_pretrained(cfg.model)
    tokenizer = get_tokenizer(cfg)

    training_args = instantiate(
        cfg.trainer, report_to="wandb" if cfg.log_wandb else None
    )

    train_dataset, validation_dataset = prepare_datasets(config=cfg)
    collator_fn = RMDataCollator(tokenizer=tokenizer,
                                 max_length=cfg.max_length)
    # Initialize our Trainer
    trainer = RMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collator_fn,
    )

    # training
    trainer.train()

    trainer.save_model(os.path.join(cfg.log_dir, f"{cfg.model.split('/')[-1]}-model"))
    tokenizer.save_pretrained(cfg.log_dir)


if __name__ == "__main__":
    
    train()




