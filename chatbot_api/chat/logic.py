import json
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 🔹 Load intent data
with open("intents.json", "r", encoding="utf-8") as f:
    intent_data = json.load(f)

intents = {k: v["examples"] for k, v in intent_data.items()}
intent_keywords = {k: v["keywords"] for k, v in intent_data.items()}
intent_responses = {k: v["response"] for k, v in intent_data.items()}

# 🔹 Load embedding model
intent_model = SentenceTransformer('all-MiniLM-L6-v2')

# 🔹 Average one embedding per intent (for multi-intent detection)
intent_embeddings = {}
for intent, phrases in intents.items():
    phrase_embeddings = intent_model.encode(phrases, convert_to_tensor=True)
    mean_embedding = torch.mean(phrase_embeddings, dim=0)
    intent_embeddings[intent] = mean_embedding

# 🔹 Keyword fallback (basic match)
def keyword_match(user_input):
    user_input = user_input.lower()
    matched = []
    for intent, keywords in intent_keywords.items():
        if any(word in user_input for word in keywords):
            matched.append(intent)
    return matched

# 🔹 Load fallback LLM (optional, not used in new fallback)
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
chat_history_ids = None

def query_llm(prompt):
    global chat_history_ids
    input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([chat_history_ids, input_ids], dim=-1) if chat_history_ids is not None else input_ids
    output_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    chat_history_ids = output_ids
    return tokenizer.decode(output_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

# 🔹 Multi-intent detection
def detect_multiple_intents(user_input, threshold=0.7):
    query_embedding = intent_model.encode(user_input, convert_to_tensor=True)
    matched = {}

    for intent, intent_emb in intent_embeddings.items():
        score = util.cos_sim(query_embedding, intent_emb).item()
        if score >= threshold:
            matched[intent] = score

    return matched  # format: {intent: score}

# 🔹 Main chatbot handler
def handle_user_input(user_input):
    # Step 1: Try intent match via embedding
    matched_intents = detect_multiple_intents(user_input)

    if matched_intents:
        reply_lines = []
        for intent in matched_intents:
            reply_lines.append(f"- {intent_responses[intent]}")
        return {
            "intent": list(matched_intents.keys()),
            "response": "\n".join(reply_lines),
            "source": "multi-intent"
        }

    # Step 2: Try keyword fallback
    keyword_matches = keyword_match(user_input)
    if keyword_matches:
        reply_lines = [f"- {intent_responses[intent]}" for intent in keyword_matches]
        return {
            "intent": keyword_matches,
            "response": "\n".join(reply_lines),
            "source": "keyword"
        }

    # Step 3: Suggest options (guided fallback)
    suggestions = "\n".join([f"• {v['examples'][0]}" for k, v in intent_data.items()])
    fallback_response = (
        suggestions
    )
    return {
        "intent": None,
        "response": fallback_response,
        "source": "suggestion"
    }
