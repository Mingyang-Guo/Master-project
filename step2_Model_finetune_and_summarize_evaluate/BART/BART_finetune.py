
# Fine_Tune_T5_train.py
# One-click fine-tune for T5 radiology summarization (NO CLI args)
# Input JSON format is aligned with RadSumBART script
# Exports final HF model dir: T5_final/

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_scheduler
from rouge import Rouge




# ==========================================================
# Path and file settings
# ==========================================================

# Path and file settings
REPO_ROOT = Path(__file__).resolve().parents[2]
PRIVATE_ROOT = REPO_ROOT / "local_private_data"
MODEL_CHECKPOINT = PRIVATE_ROOT / "models" / "facebook" / "bart-large-cnn"

# Path and file settings
DATA_DIR = PRIVATE_ROOT / "step2" / "splits"
TRAIN_JSON = DATA_DIR / "CXR_train.json"
VAL_JSON   = DATA_DIR / "CXR_val.json"
TEST_JSON  = DATA_DIR / "CXR_test.json"

# Path and file settings
OUT_DIR = PRIVATE_ROOT / "step2" / "generated" / "training" / "BART"
FINAL_HF_DIR = PRIVATE_ROOT / "models" / "BART_final"

OUT_DIR.mkdir(parents=True, exist_ok=True)
# ==========================================================


# ==========================================================
# Code path note
# ==========================================================
SEED = 5
LR = 2e-5
EPOCHS = 20
BATCH_SIZE = 16
MAX_DATASET_SIZE = 10**9
MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 128
BEAM_SIZE = 5
NO_REPEAT_NGRAM_SIZE = 2
WARMUP_STEPS = 100
# ==========================================================


def seed_everything(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def norm_text(x: str) -> str:
    if x is None:
        return ""
    x = str(x).replace("\r\n", "\n").replace("\r", "\n").strip()
    return x.strip()


def _unwrap_json(obj: Any) -> List[Dict[str, Any]]:
    """
    Turn various JSON structures into a list of dict samples.
    Supports:
      - list[dict]
      - dict[id]=dict
      - {"data":[...]} or {"train":[...]} etc.
    """
    if obj is None:
        return []

    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]

    if isinstance(obj, dict):
        for k in ["data", "train", "valid", "val", "test", "samples", "items"]:
            v = obj.get(k, None)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]

        vals = list(obj.values())
        if vals and all(isinstance(v, dict) for v in vals):
            return vals

        for v in obj.values():
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]

    return []


def load_samples(json_path: Path, max_n: int) -> List[Dict[str, str]]:
    """
    Load samples from JSON and normalize into:
      {"content": findings_text, "title": impression_text}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    raw = _unwrap_json(obj)

    in_keys = ["findings", "content", "input", "source", "src", "text", "report"]
    out_keys = ["impression", "title", "summary", "target", "tgt", "label"]

    samples: List[Dict[str, str]] = []
    for item in raw:
        inp = None
        out = None

        for k in in_keys:
            if k in item and isinstance(item[k], str):
                inp = item[k]
                break

        for k in out_keys:
            if k in item and isinstance(item[k], str):
                out = item[k]
                break

        if inp is None or out is None:
            continue

        inp = norm_text(inp)
        out = norm_text(out)

        if not inp or not out:
            continue
        # RadSumBART-compatible training input: raw Findings -> Impression.

        samples.append({"content": inp, "title": out})
        if len(samples) >= max_n:
            break

    return samples


class SumDataset(Dataset):
    def __init__(self, samples: List[Dict[str, str]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.samples[idx]


def build_collate_fn(tokenizer):
    def collate_fn(batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        batch_inputs = [x["content"] for x in batch]
        batch_targets = [x["title"] for x in batch]

        enc = tokenizer(
            batch_inputs,
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
            return_tensors="pt",
        )

        labels = tokenizer(
            text_target=batch_targets,
            padding=True,
            truncation=True,
            max_length=MAX_TARGET_LENGTH,
            return_tensors="pt",
        )["input_ids"]

        # Prepare model inputs
        labels[labels == tokenizer.pad_token_id] = -100

        enc["labels"] = labels
        return enc

    return collate_fn


@torch.no_grad()
def eval_rouge(dataloader, model, tokenizer, device) -> Dict[str, float]:
    rouge = Rouge()
    preds: List[str] = []
    refs: List[str] = []

    model.eval()
    for batch in dataloader:
        labels = batch["labels"].clone()
        batch = {k: v.to(device) for k, v in batch.items()}

        gen_ids = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_new_tokens=MAX_TARGET_LENGTH,
            num_beams=BEAM_SIZE,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
        )

        pred_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        labels[labels == -100] = tokenizer.pad_token_id
        ref_text = tokenizer.batch_decode(labels, skip_special_tokens=True)

        preds.extend([t.strip() for t in pred_text])
        refs.extend([t.strip() for t in ref_text])

    if len(preds) == 0 or len(refs) == 0:
        return {"avg": 0.0, "rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0}

    scores = rouge.get_scores(preds, refs, avg=True)

    # Compute evaluation metrics
    avg = float(scores["rouge-l"]["f"])
    return {
        "avg": avg,
        "rouge1_f": float(scores["rouge-1"]["f"]),
        "rouge2_f": float(scores["rouge-2"]["f"]),
        "rougeL_f": float(scores["rouge-l"]["f"]),
    }


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    for p in [MODEL_CHECKPOINT, TRAIN_JSON, VAL_JSON, TEST_JSON]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    seed_everything(SEED)

    # -------- load data --------
    train_samples = load_samples(TRAIN_JSON, MAX_DATASET_SIZE)
    val_samples   = load_samples(VAL_JSON, MAX_DATASET_SIZE)
    test_samples  = load_samples(TEST_JSON, MAX_DATASET_SIZE)

    print(f"[INFO] Loaded samples -> train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")
    if len(train_samples) == 0 or len(val_samples) == 0:
        raise RuntimeError(
            "Your JSON seems not parseable by this script. "
            "Please confirm it contains findings/impression or content/title fields."
        )

    train_ds = SumDataset(train_samples)
    val_ds   = SumDataset(val_samples)
    test_ds  = SumDataset(test_samples) if len(test_samples) else None

    # -------- load T5 model/tokenizer --------
    print(f"[INFO] Loading model from: {MODEL_CHECKPOINT}")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_CHECKPOINT))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_CHECKPOINT)).to(device)

    collate_fn = build_collate_fn(tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    optimizer = AdamW(model.parameters(), lr=LR)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=EPOCHS * len(train_loader),
    )

    best_avg = -1.0
    best_path: Optional[Path] = None

    # -------- train loop --------
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            total_loss += float(loss.item())
            if step % 100 == 0:
                avg_loss = total_loss / step
                print(f"[TRAIN] epoch={epoch} step={step}/{len(train_loader)} loss={avg_loss:.4f}")

        # -------- validation rouge --------
        valid_scores = eval_rouge(val_loader, model, tokenizer, device)
        rouge_avg = valid_scores["avg"]
        print(
            f"[VALID] epoch={epoch} ROUGE_avg={rouge_avg:.4f} "
            f"(R1={valid_scores['rouge1_f']:.4f}, "
            f"R2={valid_scores['rouge2_f']:.4f}, "
            f"RL={valid_scores['rougeL_f']:.4f})"
        )

        epoch_bin = OUT_DIR / f"epoch_{epoch}_rouge_{rouge_avg:.4f}.bin"
        torch.save(model.state_dict(), str(epoch_bin))
        print(f"[OK] Saved epoch weights -> {epoch_bin}")

        if rouge_avg > best_avg:
            best_avg = rouge_avg
            best_path = OUT_DIR / f"BEST_epoch_{epoch}_valid_rouge_{rouge_avg:.4f}_model_weights.bin"
            torch.save(model.state_dict(), str(best_path))
            print(f"[OK] New BEST saved -> {best_path}")

    if best_path is None:
        raise RuntimeError("No best weights saved.")

    # -------- export HF final dir --------
    print("[EXPORT] Exporting final HuggingFace model dir...")
    export_model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_CHECKPOINT))
    export_model.load_state_dict(torch.load(str(best_path), map_location="cpu"))
    export_model.eval()

    FINAL_HF_DIR.mkdir(parents=True, exist_ok=True)
    export_model.save_pretrained(str(FINAL_HF_DIR), safe_serialization=True)
    tokenizer.save_pretrained(str(FINAL_HF_DIR))

    meta = {
        "base_model_checkpoint": str(MODEL_CHECKPOINT),
        "train_json": str(TRAIN_JSON),
        "val_json": str(VAL_JSON),
        "test_json": str(TEST_JSON),
        "best_weights": str(best_path),
        "best_valid_rouge_avg": best_avg,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "max_input_length": MAX_INPUT_LENGTH,
        "max_target_length": MAX_TARGET_LENGTH,
        "beam_size": BEAM_SIZE,
        "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,
    }

    with open(OUT_DIR / "training_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Final model exported -> {FINAL_HF_DIR}")
    print(f"[OK] Meta saved -> {OUT_DIR / 'training_meta.json'}")
    print("===== DONE =====")


if __name__ == "__main__":
    main()
