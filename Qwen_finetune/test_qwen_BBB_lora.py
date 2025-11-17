import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
import re


MODEL_DIR = "./qwen2_7B_bbbp_lora"
use_few_shot = True

DATA_PATH = "data/few_shot_examples.csv"      # few-shot examples
TEST_DATA_PATH = "data/BBBP_test_data.csv"   # test data

MAX_LENGTH = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_few_shot_prompt(smiles_query, df_examples=None, k=4):
    """
    Build a chat-style prompt with:
    - BBB background message
    - k few-shot examples (if provided)
    - query molecule
    Output format is JSON: {"answer": 0 or 1}
    """
    bbb_background = (
        "The blood-brain barrier (BBB) is a selective barrier that restricts the passage "
        "of most molecules from the bloodstream into the brain. BBB-penetrant molecules "
        "are often moderately lipophilic, relatively small, and avoid strong ionization "
        "and extensive hydrogen bonding at physiological pH. Highly polar or large "
        "molecules usually show poor BBB permeability."
    )

    system_msg = (
        "You are an expert medicinal chemist. "
        "You classify molecules as BBB-penetrant (1) or non-penetrant (0). "
        "Use your knowledge of the BBB and any labeled examples provided."
    )

    # Base query
    user_query = (
        "Analyze the following molecule and predict whether it can cross the "
        "blood-brain barrier (BBB).\n\n"
        f"SMILES: {smiles_query}\n\n"
        "Respond ONLY with a JSON object in the following format:\n\n"
        '{\n'
        '  "answer": <answer>\n'
        '}\n\n'
        "Use 0 for non-BBB-penetrant and 1 for BBB-penetrant. Do not include any "
        "additional text, explanations, or comments outside the JSON object.\n"
    )

    if df_examples is not None:
        examples_text = []
        # Expect columns "SMILES" and "Label" in the few-shot CSV
        for _, row in df_examples.iterrows():
            smiles = row["SMILES"]
            label = int(row["Label"])
            examples_text.append(
                "Example:\n"
                f"SMILES: {smiles}\n"
                f'Output JSON: {{"answer": {label}}}\n'
            )
        few_shot_block = "\n".join(examples_text)
        prompt = (
            f"<|system|>\n{system_msg}\n\n"
            f"Background about BBB:\n{bbb_background}\n\n"
            "<|user|>\n"
            "Here are some labeled examples:\n\n"
            f"{few_shot_block}\n\n"
            f"{user_query}"
            "<|assistant|>\n"
        )
    else:
        prompt = (
            f"<|system|>\n{system_msg}\n\n"
            f"Background about BBB:\n{bbb_background}\n\n"
            "<|user|>\n"
            f"{user_query}"
            "<|assistant|>\n"
        )
    return prompt


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    model.eval()
    return model.to(DEVICE), tokenizer


def parse_json_answer(gen_text: str):
    """
    Extract {"answer": 0/1} from the model's output.
    Returns (prediction:int|None, json_str:str|None).
    """
    # Find the first {...} block
    match = re.search(r"\{.*?\}", gen_text, re.DOTALL)
    if not match:
        return None, None

    json_str = match.group(0)
    try:
        obj = json.loads(json_str)
    except json.JSONDecodeError:
        return None, json_str

    ans = obj.get("answer", None)
    if ans in (0, 1, "0", "1"):
        return int(ans), json_str
    else:
        return None, json_str


def predict_bbb(smiles: str, model, tokenizer, df_examples: pd.DataFrame | None, k: int = 4):
    # If few-shot is enabled and df_examples is not None, pass head(k); else pass None
    examples = df_examples.head(k) if (df_examples is not None and use_few_shot) else None
    prompt = build_few_shot_prompt(smiles, examples, k=k)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(DEVICE)

    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,          # allow enough tokens for JSON
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_ids = outputs[0][input_len:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # First try to parse JSON
    prediction, json_str = parse_json_answer(gen_text)

    # Fallback: if JSON parsing fails, try to extract first '0' or '1' character
    if prediction is None:
        for ch in gen_text:
            if ch in ("0", "1"):
                prediction = int(ch)
                break

    # For logging/debugging, we return the full generation (which should be JSON)
    return prediction, gen_text


def main():
    # Load model + tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # Few-shot examples (optional)
    if use_few_shot:
        df = pd.read_csv(DATA_PATH)
        df["Label"] = df["Label"].astype(int)
    else:
        df = None

    # Test data
    test_df = pd.read_csv(TEST_DATA_PATH)
    test_df["True_Label"] = test_df["True_Label"].astype(int)

    predictions = []
    for _, row in test_df.iterrows():
        smiles = row["SMILES"]
        pred_label, raw_output = predict_bbb(smiles, model, tokenizer, df_examples=df, k=8)
        predictions.append(
            {
                "SMILES": smiles,
                "True_Label": row["True_Label"],
                "Predicted_Label": pred_label,
                "Raw_Output": raw_output,
            }
        )

    predictions_df = pd.DataFrame(predictions)
    os.makedirs("predictions", exist_ok=True)
    predictions_df.to_csv(
        f"predictions/qwen2_7B_bbbp_lora_8_shots_predictions.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
