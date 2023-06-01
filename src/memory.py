from deepspeed.runtime.zero.stage_1_and_2 import (
    estimate_zero2_model_states_mem_needs_all_live,
)
from transformers import AutoModel

model = AutoModel.from_pretrained("togethercomputer/RedPajama-INCITE-Base-3B-v1")
print(
    estimate_zero2_model_states_mem_needs_all_live(
        model, num_gpus_per_node=2, num_nodes=1
    )
)
