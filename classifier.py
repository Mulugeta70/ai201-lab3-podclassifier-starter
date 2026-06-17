import json
import os
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, VALID_LABELS, DATA_PATH, TRAIN_FILE, LABELS_FILE

_client = Groq(api_key=GROQ_API_KEY)


def load_labeled_examples() -> list[dict]:
    """
    Load the training episodes and merge them with the student's labels.

    Returns a list of dicts, each with:
      - "id"          : episode ID
      - "title"       : episode title
      - "podcast"     : podcast name
      - "description" : episode description
      - "label"       : the label from my_labels.json (may be None if not yet annotated)

    Only returns episodes where the label is a valid, non-null string.
    Episodes with null labels are silently skipped.
    """
    train_path = os.path.join(DATA_PATH, TRAIN_FILE)
    labels_path = os.path.join(DATA_PATH, LABELS_FILE)

    with open(train_path, encoding="utf-8") as f:
        episodes = {ep["id"]: ep for ep in json.load(f)}

    with open(labels_path, encoding="utf-8") as f:
        labels = {entry["id"]: entry["label"] for entry in json.load(f)}

    labeled = []
    for ep_id, ep in episodes.items():
        label = labels.get(ep_id)
        if label in VALID_LABELS:
            labeled.append({**ep, "label": label})

    return labeled


def build_few_shot_prompt(labeled_examples: list[dict], description: str) -> str:
    """
    Build a few-shot classification prompt using the student's labeled training examples.
    """
    task_instruction = (
        "You are classifying podcast episodes by their structural format.\n"
        "Classify the episode into exactly one of these four labels:\n\n"
        "- interview: One host plus one guest. The host asks questions; the guest answers. "
        "The guest's expertise, experience, or story drives the episode.\n"
        "- solo: One host speaking alone — no guests. Personal reflection, opinion, or analysis. "
        "The host is the only source, speaking from their own voice and memory.\n"
        "- panel: Three or more speakers as rough equals — a roundtable or group discussion. "
        "No single person is being interviewed.\n"
        "- narrative: A reported or documentary story assembled from external sources — documents, "
        "archives, other people's interviews — with a clear narrative arc.\n\n"
        "Respond in exactly this format and nothing else:\n"
        "Label: <interview|solo|panel|narrative>\n"
        "Reasoning: <one sentence explaining why>\n"
    )

    examples_block = "\nHere are labeled examples to learn the pattern from:\n"
    for ex in labeled_examples:
        examples_block += (
            f"\n---\n"
            f"Title: {ex['title']}\n"
            f"Description: {ex['description']}\n"
            f"Label: {ex['label']}\n"
        )

    new_episode = (
        "\n---\n"
        "Now classify this episode:\n\n"
        f"Description: {description}\n\n"
        "Label: ?"
    )

    return task_instruction + examples_block + new_episode


def classify_episode(description: str, labeled_examples: list[dict]) -> dict:
    """
    Classify a single podcast episode description using the few-shot LLM classifier.
    Returns a dict with "label" (one of VALID_LABELS or "unknown") and "reasoning".
    """
    try:
        prompt = build_few_shot_prompt(labeled_examples, description)

        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        text = response.choices[0].message.content.strip()

        label = "unknown"
        reasoning = text

        for line in text.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("label:"):
                candidate = stripped.split(":", 1)[1].strip().lower()
                if candidate in VALID_LABELS:
                    label = candidate
            elif stripped.lower().startswith("reasoning:"):
                reasoning = stripped.split(":", 1)[1].strip()

        return {"label": label, "reasoning": reasoning}

    except Exception as e:
        return {"label": "unknown", "reasoning": f"Classification failed: {e}"}
