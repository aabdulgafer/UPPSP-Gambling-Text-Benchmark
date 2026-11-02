import re , time, torch
import numpy as np
from transformers import AutoTokenizer, PreTrainedModel
from collections import defaultdict
from typing import Dict, Any, List, Callable
from .prompts_M import engagingness_system_prompt_M, naturalness_system_prompt_M, coherence_system_prompt_M, groundedness_system_prompt_M

device = "cuda" if torch.cuda.is_available() else "cpu"


def calculate_metrics_M(sentences: List[str], metric_funcs: Dict[str, Callable[..., Any]]) -> Dict[Any, Any]:
    """
    calculates metrics which evaluate the monologue quality.
    """
    metric_scores = defaultdict(dict)
    for metric_name, metric_fn in metric_funcs.items():
        try:
            metric_scores[metric_name] = metric_fn(sentences)
            print(">>>>> cooloff for 5 seconds")
            time.sleep(5)
        except Exception as e:
            print(f' =>Ran into error {e} while processing for metric {metric_name}, moving on...')
    return metric_scores


def g_eval_M(monologues: List[str], model: PreTrainedModel, tokenizer: AutoTokenizer) -> List[Dict[str, float]]:
    """
    Calculates g-eval metrics for monologues using monologue-specific prompts.
    Returns a list of dictionaries with metric scores for each monologue.
    """
    # Decide where to put the input IDs:
    # use the device of the input embedding weights (works for single-device and device_map="auto")
    try:
        embed_device = model.get_input_embeddings().weight.device
    except Exception:
        # Fallback if something weird happens
        embed_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    def g_eval_util_M(monologue: str, metric_name: str) -> float:

        score_ids = np.array([tokenizer.encode(str(i))[-1] for i in range(1, 4)])
        prompt = tokenizer.apply_chat_template(
            conversation=[
                {"role": "system", "content": monologue_metric_prompts[metric_name]},
                {"role": "user", "content": monologue}
            ],
            tokenize=False,
            add_generation_prompt=True
        )

        tokens = torch.tensor([tokenizer.encode(prompt)], device=embed_device)
        
        with torch.inference_mode():
            logits = model(input_ids=tokens).logits

        logits = logits.cpu().numpy()[0, -1, :]
        score_probs = torch.softmax(torch.tensor(logits[score_ids]), dim=-1).numpy()
        rating = (score_probs * np.arange(start=1, stop=4)).sum(0)

        return float(rating)

    monologue_metric_prompts = {
        "engagingness": engagingness_system_prompt_M,
        "naturalness": naturalness_system_prompt_M,
        "coherence": coherence_system_prompt_M,
        "groundedness": groundedness_system_prompt_M
    }

    scores = []
    count=1
    for monologue in monologues:
        metric_dict = {}
        print(f" Calculating G-Eval for monologue #{count}")
        for metric in monologue_metric_prompts:
            metric_dict[metric] = g_eval_util_M(monologue, metric)
        scores.append(metric_dict)
        print(f"Monologue {count} evaluated")
        if count % 200 == 0:
            print("Cooling off for 5s after 200 evaluations...")
            time.sleep(5)

        count+=1

    return scores