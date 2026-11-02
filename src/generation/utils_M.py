import ast, re, time, torch, evaluate, os, random,json
import pandas as pd
from typing import List, Dict
from transformers import AutoTokenizer
from torch import inference_mode
from .prompts_M import generate_topics_prompt, generate_user_profiles_prompt, generate_Twitter_monologue_prompt
from .prompts_M import judge_prompt_M 
from datetime import datetime, timedelta, timezone
from .config_M import (
    EXPLICIT_PROFILE_RATIO, SEED, MIN_DIMENSIONS_PER_MONOLOGUE,OPENER_MAX_WORDS, 
    DIMENSIONS_SHORT, DIMENSION_TO_STATEMENT,
    USE_REFLECTION_EXAMPLES_FOR_OP, REFLECTION_EXAMPLES_PER_PROMPT, REFLECTION_AREAS,
    MAX_EURO_MENTIONS_PER_MONOLOGUE, MAX_BRAND_MENTIONS_PER_MONOLOGUE,
    ALLOW_BRAND_FROM_EXPLICIT_ONLY,EMOJI_MIN, EMOJI_MAX, OPENER_MIN_WORDS, PROFILE_MAX_TOKENS
)
random.seed(SEED)

# ----------------- Runtime / tokenizer setup -----------------
#stop_tokens = ["<|eot_id|>"]
stop_tokens = ["<|end|>", "<|endoftext|>", "<|eot_id|>"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#n_gpu_layers = 32 if device.type == "cuda" else 0
#print(f'Running on device: {device}, layers offloaded: {n_gpu_layers}')
#print(f"PyTorch device detected: {device}", flush=True)
print(
    f"PyTorch CUDA available: {torch.cuda.is_available()} "
    "(not used to control llama.cpp GPU offloading)",
    flush=True,
)

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="microsoft/Phi-3.5-mini-instruct", trust_remote_code=True)
#rouge = evaluate.load('rouge')

# ----------------- Helpers -----------------
def calculate_rouge_l(topic1: str, topic2: str) -> float:
    score = rouge.compute(references=[topic2], predictions=[topic1])
    return score['rougeL']

def filter_repeated_topics(topics: List[str], threshold: float = 0.5) -> List[str]:
    unique_topics = []
    for topic in topics:
        if all(calculate_rouge_l(topic, u) < threshold for u in unique_topics):
            unique_topics.append(topic)
    return unique_topics

#Paper 2
# --- Personality helpers (Big Five) ---
import json as _json

def _labelize_uppsp(v):
    v = float(v)
    return "high" if v >= 3.0 else ("low" if v < 2.0 else "mid")


UPPSP_KEYS = [
    "negative_urgency",
    "positive_urgency",
    "lack_of_premeditation",
    "lack_of_perseverance",
    "sensation_seeking",
]

FORBIDDEN_PROFILE_TERMS = {
    "addiction",
    "addicted",
    "gambling addiction",
    "problem gambling",
    "diagnosis",
    "disorder",
    "therapy",
    "relapse",
    "recovery journey",
    "ruined my life",
    "lost everything",
    "hit rock bottom",
}

def _has_forbidden_profile_terms(text: str) -> bool:
    s = (text or "").lower()
    return any(term in s for term in FORBIDDEN_PROFILE_TERMS)

def _verbalize_traits(traits_json: str) -> str:
    try:
        t = _json.loads(traits_json) if isinstance(traits_json, str) else traits_json
        L = t.get("labels") or {}
    except Exception:
        L = {}

    lines = []

    if L.get("negative_urgency") == "high":
        lines.append("negative_urgency=high: impulsive reactions under stress or bad mood")
    elif L.get("negative_urgency") == "mid":
        lines.append("negative_urgency=mid: occasional emotional impulsivity under stress")
    elif L.get("negative_urgency") == "low":
        lines.append("negative_urgency=low: controlled when upset")

    if L.get("positive_urgency") == "high":
        lines.append("positive_urgency=high: impulsive reactions when excited or winning")
    elif L.get("positive_urgency") == "mid":
        lines.append("positive_urgency=mid: some impulsivity during positive excitement")
    elif L.get("positive_urgency") == "low":
        lines.append("positive_urgency=low: controlled even when excited")

    if L.get("lack_of_premeditation") == "high":
        lines.append("lack_of_premeditation=high: acts without thinking through consequences")
    elif L.get("lack_of_premeditation") == "mid":
        lines.append("lack_of_premeditation=mid: sometimes plans, sometimes acts quickly")
    elif L.get("lack_of_premeditation") == "low":
        lines.append("lack_of_premeditation=low: thinks carefully before acting")

    if L.get("lack_of_perseverance") == "high":
        lines.append("lack_of_perseverance=high: struggles to stick with limits or finish intentions")
    elif L.get("lack_of_perseverance") == "mid":
        lines.append("lack_of_perseverance=mid: inconsistent follow-through")
    elif L.get("lack_of_perseverance") == "low":
        lines.append("lack_of_perseverance=low: usually follows through")

    if L.get("sensation_seeking") == "high":
        lines.append("sensation_seeking=high: seeks thrill, novelty, or risky excitement")
    elif L.get("sensation_seeking") == "mid":
        lines.append("sensation_seeking=mid: moderate interest in excitement")
    elif L.get("sensation_seeking") == "low":
        lines.append("sensation_seeking=low: avoids unnecessary risk or thrill")

    return ". ".join(lines) or "balanced UPPS-P profile"


# 1 ----------------- Subtopics -----------------

def normalize(s):
    s = re.sub(r"\s+", " ", s.strip().lower())
    s = re.sub(r"[^\w\s]", "", s)
    return s

def generate_sub_topics(model, topic: str, n_sub_topics: int) -> List[str]:
    """
    Generate <= n_sub_topics implicit, diverse, non-duplicate subtopics for a given topic.
    - Enforces implicitness (bans explicit gambling/clinical terms in SUBTOPICS).
    - Enforces opener diversity (first 2 words unique within this topic).
    - Prunes near-duplicates via char-5gram Jaccard (no sklearn).
    - Caps length to <= 10 words (tweety, concise).
    """
    assert n_sub_topics <= 20, "let's keep it down to 20 for now"

    # ---------- helpers ----------
    def _too_explicit(s: str) -> bool:
        banned = {
            "addiction","addict","disorder","therapy","diagnosis",
            "casino","bet "," bet", "bets","betting","wager",
            "slots","slot","blackjack","poker","roulette","parlay"
        }
        ss = " " + s.lower() + " "  # pad to catch word boundaries for 'bet'
        return any(w in ss for w in banned)

    def _trim_words(s: str, max_words: int = 10) -> str:
        toks = re.findall(r"[^\s]+", s.strip())
        if len(toks) <= max_words:
            return s.strip()
        return " ".join(toks[:max_words]) + "…"

       
    def _opener2(s: str) -> str:
        toks = re.findall(r"[A-Za-z']+", s.lower())
        return " ".join(toks[:2]) if toks else ""

    def _char_ngrams(s: str, n: int = 5) -> set:
        s = re.sub(r"\s+", " ", s.strip().lower())
        if len(s) < n:
            return {s}
        return {s[i:i+n] for i in range(len(s)-n+1)}

    def _jaccard(a: str, b: str, n: int = 5) -> float:
        A, B = _char_ngrams(a, n), _char_ngrams(b, n)
        if not A and not B:
            return 0.0
        return len(A & B) / max(1, len(A | B))

    # ---------- call the model ----------
    # Ask for more than we need, so we can filter aggressively.
    ask = max(n_sub_topics * 4, n_sub_topics + 4)  # e.g., 8 when n=2
    prompt = tokenizer.apply_chat_template(
        conversation=[
            {"role": "system", "content": generate_topics_prompt.format(topic=topic, n_sub_topics=ask)},
            {"role": "user", "content": f"topic - {topic}"}
        ],
        tokenize=False, add_generation_prompt=True
    )
    output = model(prompt=prompt, temperature=0.6, stop=stop_tokens, max_tokens=4096, echo=False)
    raw = output['choices'][0]['text']

    # ---------- parse ----------
    match = re.search(r'\[[^\[\]]+\]', raw, re.DOTALL)
    if not match:
        print(" No valid list found in model output!")
        candidates = []
    else:
        try:
            lst = ast.literal_eval(match.group(0))
            # flatten weird cases and coerce to str
            candidates = [str(x).strip() for x in lst if str(x).strip()]
        except Exception as e:
            print(f" Failed to parse topic list: {e}")
            print(f" Failed to parse output for topic: {repr(topic)}")
            candidates = []

    # ---------- filtering pipeline ----------
    # 1) implicitness
    candidates = [c for c in candidates if not _too_explicit(c)]
    # 2) remove exact dupes
    seen = set()
    uniq = []
    for c in candidates:
        key = normalize(c)
        if key not in seen:
            seen.add(key)
            uniq.append(c)

    # 3) opener diversity (first 2 words)
    seen_openers = set()
    diverse = []
    for c in uniq:
        op = _opener2(c)
        if op and op in seen_openers:
            continue
        seen_openers.add(op)
        diverse.append(c)
 
    # 4) near-duplicate pruning via char-5gram Jaccard
    pruned = []
    for c in diverse:
        if all(_jaccard(c, p, n=5) < 0.80 for p in pruned):  # threshold can be tweaked
            pruned.append(c)

    # 5) length cap (tweety style)
    final = [_trim_words(c, max_words=10) for c in pruned]

    # 6) keep only n_sub_topics
    final = final[:n_sub_topics]

    # ---------- write & report ----------
    df = pd.DataFrame({"Topic": [topic] * len(final), "Subtopic": final})
    df.to_csv("1_Generated_Subtopics.csv", index=False, mode='a',
              header=not os.path.exists("1_Generated_Subtopics.csv"))

    if len(final) < n_sub_topics:
        print(f" =>Warning: only {len(final)} subtopics generated for '{topic}' (expected {n_sub_topics}).")

    return final

#=======================================================================================
# 2 ----------------- Extended Profiles -----------------

# ----- helpers -----
TONES = {"angry","sarcastic","ashamed","defensive","hopeful","bitter","anxious","jaded","helpful","analytical","relieved","steady","calm","proud"}
GAMES = {"sportsbook","slots","live_casino","blackjack","roulette","poker","crash","lootboxes","scratchers"}
MONEY = {"stakes_small","stakes_mid","stakes_high"}
RISK  = {"none","low","moderate","high"}
GENDERS   = {"male", "female"}

CANONICAL_BRANDS = {"888","Stake","Bet365","Betfair","WilliamHill","PaddyPower",
                    "Unibet","LuckySpin","QuickBet","SlotKing" }

GAMBLER_BEH = { "late-night ATM","chasing small losses","raises limits",
                "deletes app then re-installs","blames RNG","maxed card" }

SUPPORT_BEH = { "lurks and asks questions","posts resources","keeps budget spreadsheet",
                "tags friends to advise","celebrates no-bet streak" }

SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]*\|>")
PARENS_TRAIL_RE = re.compile(r"\s*\([^)]*\)\s*$")


def normalize_brand_name(text: str):
    key = re.sub(r"\s+", "", (text or "")).lower()
    _CANON_MAP = { re.sub(r"\s+", "", b).lower(): b for b in CANONICAL_BRANDS }
    return _CANON_MAP.get(key)

def _normalize_brand_focus(brand_focus_raw: str) -> str:

    items = _parse_list_cell(brand_focus_raw)
    out = []
    for b in items:
        canon = normalize_brand_name(b)
        if canon and canon not in out:
            out.append(canon)
        if len(out) == 3:
            break
    return ",".join(out) if out else "none"

def _brand_regex_from_canonical():
    parts = []
    for b in CANONICAL_BRANDS:
        tokens = re.findall(r"\d+|[A-Z][a-z]*", b)
        unspaced = "".join(tokens).lower()
        spaced   = r"\s+".join([t.lower() for t in tokens]) if len(tokens) > 1 else ""
        if spaced:
            parts.append(f"(?:{unspaced}|{spaced})")
        else:
            parts.append(unspaced)
    pat = "|".join(parts)
    return re.compile(rf"(?<!\w)(?:{pat})(?!\w)", re.I)

BRAND_RE = _brand_regex_from_canonical()


def _parse_list_cell(s: str) -> List[str]:
    if not s:
        return []
    # split on comma, keep phrases
    return [x.strip() for x in s.split(",") if x.strip()]

def _norm_beh(s: str) -> str:
    # normalize for matching: lowercase, collapse spaces/hyphens
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def _validate_behaviors(beh_list, risk_level: str):
    # Accept str or list, normalize to list
    if isinstance(beh_list, str):
        items = [x.strip() for x in beh_list.split(",") if x.strip()]
    else:
        items = list(beh_list or [])

    # Allowed pool
    risk = (risk_level or "none").strip().lower()
    pool = SUPPORT_BEH if risk == "none" else (GAMBLER_BEH | SUPPORT_BEH)
    pool_map = {_norm_beh(b): b for b in pool}

    out, seen = [], set()
    for b in items:
        key = _norm_beh(b)
        if key in pool_map and key not in seen:
            out.append(pool_map[key])
            seen.add(key)
            if len(out) == 2:  # hard cap 2
                break

    # Ensure at least 1
    if len(out) < 1:
        remaining = [v for k, v in pool_map.items() if k not in seen]
        random.shuffle(remaining)
        if remaining:
            out.append(remaining[0])

    return out[:2]  # 1–2


def _validate_emotions(emotions_list):
    # Accept str or list, normalize to list
    if isinstance(emotions_list, str):
        raw = [x.strip() for x in emotions_list.split(",") if x.strip()]
    else:
        raw = list(emotions_list or [])

    toks = []
    seen = set()
    for e in raw:
        t = re.sub(r'[^a-z]', '', str(e).lower())
        if t and t not in seen:
            toks.append(t)
            seen.add(t)
            if len(toks) == 2:  # hard cap 2
                break

    if len(toks) < 1:
        toks = ["anxious"]  # default ensures >=1

    return toks[:2]  # 1–2


def _strip_special(s: str) -> str:
    s = (s or "").strip()
    s = SPECIAL_TOKEN_RE.sub("", s)          # remove <|...|>
    s = PARENS_TRAIL_RE.sub("", s)           # drop trailing (...) notes
    s = s.replace("|", "/")                   # avoid pipe leaks
    return " ".join(s.split())                # collapse spaces


# ----------------------------------
# Generate a clean, natural user bio (PASS-THROUGH)
# ----------------------------------
def make_profile_description(profile: Dict[str, str]) -> str:
    # keep what the model wrote; remove artifacts only
    desc = (profile.get("description") or "").strip()
    return _strip_special(desc)

# ----------------------------------
# Generate full profile line for monologue prompt (MATCH CSV COLUMNS)
# ----------------------------------
def make_profile_line(profile: Dict[str, str]) -> str:
    import ast

    def _tolist(val):
        if isinstance(val, list):
            return [x for x in val if str(x).strip()]
        s = (val or "").strip()
        if not s:
            return []
        # try "['a','b']"
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
        # fallback "a, b"
        return [p.strip(" '\"") for p in s.split(",") if p.strip(" '\"")]

    def _stakes_phrase(v: str) -> str:
        v = (v or "").strip().lower()
        return {"stakes_small":"small stakes", "stakes_mid":"mid stakes", "stakes_high":"high stakes"}.get(v, "")

    user_id = profile.get("user_id", "U0000")

    # 1) base bio: description from CSV (strip outer quotes + trailing dot)
    base = (profile.get("description") or "").strip()
    if base.startswith('"') and base.endswith('"') and len(base) >= 2:
        base = base[1:-1]
    base = base.rstrip(".")

    # 2) pull EXACT columns (with fallbacks for old names)
    tone   = (profile.get("tone_baseline") or profile.get("tone_base") or "").strip().lower()
    money  = (profile.get("money_focus") or "").strip().lower()
    game   = (profile.get("game_focus")  or "").strip().lower()
    emos   = _tolist(profile.get("emotional_profile") or profile.get("emotions") or "")
    behs   = _tolist(profile.get("behavior_traits")  or profile.get("behaviors") or "")
    risk = (profile.get("risk_level") or "none").strip().lower()

    # 3) build compact tail strictly from these fields
    parts = []

    # Most important conditioning first
    parts.append(f"Gambling risk level: {risk}")

    # personality (compact, human-friendly)
    pj = profile.get("personality_traits", "")
    trait_desc = _verbalize_traits(pj)
    if trait_desc:
        parts.append(f"Personality traits: {trait_desc}")

    # Then style/context details
    if tone:
        parts.append(f"{tone} tone")

    if emos:
        emos2 = [e for e in emos if e and e.lower() != tone][:2]
        if emos2:
            parts.append("feels " + ", ".join(emos2))

    if game and game != "none":
        parts.append(f"plays {game}")

    sp = _stakes_phrase(money)
    if sp:
        parts.append(sp)

    if behs:
        parts.append(f"tends to {behs[0]}")

    #line = f'{user_id}: "{base}"'
    line = f'{user_id}: background note, lower priority than risk/personality: "{base}"'
    if parts:
        line += "; " + "; ".join(parts) + "."
    else:
        line += "."

    return line


'''
    # 3) build compact tail strictly from these fields
    parts = []
    parts.append(f"risk level: {risk}")
    if tone:
        parts.append(f"{tone} tone")
    if emos:
        # up to 2 emotions, avoid duplicating the tone word
        emos2 = [e for e in emos if e and e.lower() != tone][:2]
        if emos2:
            parts.append("feels " + ", ".join(emos2))
    if game and game != "none":
        parts.append(f"plays {game}")
    sp = _stakes_phrase(money)
    if sp:
        parts.append(sp)
    if behs:
        parts.append(f"tends to {behs[0]}")

    line = f'{user_id}: "{base}"'
    if parts:
        line += "; " + "; ".join(parts) + "."
    else:
        line += "."
    # personality (compact, human-friendly)
    pj = profile.get("personality_traits","")
    trait_desc = _verbalize_traits(pj)
    if trait_desc:
        line += f" Personality traits: {trait_desc}."
    
    return line
'''
# ----------------------------------
# Create full prompt block for monologues
# ----------------------------------
def make_profiles_block(profiles: List[Dict[str, str]]) -> str:
    return "\n".join(make_profile_line(p) for p in profiles)


def normalize_profile(p: dict) -> dict:
    # risk basic
    rl = (p.get("risk_level") or "").strip().lower() or "none"
    p["risk_level"] = rl

    # sanitize bio minimally (no rewriting, no truncation)
    desc = _strip_special(str(p.get("description", ""))).strip()
    p["description"] = desc

    pt_raw = p.get("personality_traits", "")

    # just ensure it is valid JSON; if not → leave original or synthesize fallback
    try:
        pt = json.loads(pt_raw)  # loads → ensures it's valid
        # dump back to normalized JSON string, but keep structure intact
        p["personality_traits"] = json.dumps(pt, ensure_ascii=False)
    except Exception:
        # fallback: leave as-is or replace with synthetic UPPSP JSON
        p["personality_traits"] = pt_raw

    return p


def _clean_enums(age, gender,risk_level, tone_baseline, money_focus, game_focus):
    # age
    try:
        age = int(age)
    except Exception:
        age = random.randint(18, 70)
    age = max(18, min(70, age))

    gender = (gender or "").strip().lower() or "male"
    risk_level = (risk_level or "").strip().lower() or "none"
    tone_baseline = (tone_baseline or "").strip().lower()
    money_focus = (money_focus or "").strip().lower()
    game_focus = (game_focus or "").strip().lower()

    if gender not in GENDERS: gender="male"
    if risk_level not in RISK: risk_level = "none"
    #if tone_baseline not in TONES: tone_baseline = random.choice(TONES)
    #random.choice(list(TONES))
    if tone_baseline not in TONES:
        tone_baseline = random.choice(list(TONES))
    if game_focus not in GAMES: game_focus = random.choice(list(GAMES))
    if money_focus not in MONEY: money_focus = "stakes_small"

    return age, gender, risk_level, tone_baseline, money_focus, game_focus


def generate_profiles(model,  n_profiles: int, explicit_ratio: float = EXPLICIT_PROFILE_RATIO) -> List[Dict]:

    assert n_profiles <= 3000, "let's keep it down to 3000 for now"

    RNG = random.Random(SEED)       

    def _strip_listish(brand_focus, emotions, behaviors):
        brand_focus  = (brand_focus or "").strip().strip('"').strip("'")
        emotions     = (emotions or "").strip().strip('"').strip("'")
        behaviors    = (behaviors or "").strip().strip('"').strip("'")
        return brand_focus, emotions, behaviors


    def _repair_or_default(parts: List[str]) -> List[str]:
        """Ensure we have exactly 12 fields; pad/trim sensibly."""
        parts = [ (p or "").strip() for p in parts ]
        if len(parts) < 12:
            parts += [""] * (12 - len(parts))
        elif len(parts) > 12:
            # merge overflow into description (last field)
            head = parts[:11]
            tail = " | ".join(parts[11:])
            parts = head + [tail]
        return parts

    def _coerce_record(parts: List[str]) -> Dict:
        parts = _repair_or_default(parts)
        (user_id, age, gender, risk_level, tone_baseline, money_focus, brand_focus, game_focus,
         emotions, behaviors, description, personality_traits) = parts

        brand_focus, emotions, behaviors = _strip_listish(brand_focus, emotions, behaviors)
        age,gender,risk_level, tone_baseline, money_focus, game_focus = _clean_enums(age,gender,risk_level, tone_baseline, money_focus, game_focus)
        brand_focus = _normalize_brand_focus(brand_focus)
        emo_list = _validate_emotions(_parse_list_cell(emotions))
        emotions = ", ".join(emo_list)
        beh_list = _validate_behaviors(_parse_list_cell(behaviors), risk_level)
        behaviors = ", ".join(beh_list)

        # backfill empties with minimal plausible content
        if not user_id: user_id = "UX"
        if not gender: gender = "male"
        if not emotions: emotions = "anxious, frustrated"
        if not description: description = "Trying to keep it together, not always great at it."          
        
        try:
            pj = json.loads(personality_traits) if personality_traits else {}
            vals = pj.get("values") or {}
            # sanity
            labels = pj.get("labels") or {}
            for k in UPPSP_KEYS:
                if labels.get(k) not in {"low", "mid", "high"}:
                     labels[k] = _labelize_uppsp(float(vals[k]))

            pj["labels"] = labels
            
        except Exception:
            # synthesize if missing/bad
            import random as _rnd
            def _bounded(mu=0.5, sigma=0.18):
                import numpy as _np
                return float(_np.clip(_np.random.normal(mu, sigma), 0.0, 1.0))

            
            def _rand_uppsp():
                return round(random.uniform(1.0, 4.0), 2)

            nu = _rand_uppsp()
            pu = _rand_uppsp()
            lprem = _rand_uppsp()
            lper = _rand_uppsp()
            ss = _rand_uppsp()

            pj = {
                "model": "UPPS-P",
                #"version": "1.0",
                "values": {
                    "negative_urgency": nu,
                    "positive_urgency": pu,
                    "lack_of_premeditation": lprem,
                    "lack_of_perseverance": lper,
                    "sensation_seeking": ss
                },
                    "labels": {
                        "negative_urgency": _labelize_uppsp(nu),
                        "positive_urgency": _labelize_uppsp(pu),
                        "lack_of_premeditation": _labelize_uppsp(lprem),
                        "lack_of_perseverance": _labelize_uppsp(lper),
                        "sensation_seeking": _labelize_uppsp(ss)
                    }
                }
               
        desc = _strip_special(description or "").strip()
        if desc.lower().startswith("description:"):
            desc = re.sub(r"^description\s*:\s*", "", desc, flags=re.I).strip()

        if re.search(r"\{.*UPPS-P.*\}", desc, flags=re.I):
            raise ValueError(f"Description contains personality JSON: {desc}")

        if re.search(r"\bU\d{4}\b", desc):
             raise ValueError(f"Description contains user id: {desc}")

        if len(desc.split()) < 5:
            raise ValueError(f"Description too short/corrupted: {desc}")
        
        if _has_forbidden_profile_terms(desc):
            raise ValueError(f"Forbidden clinical/profile term in description: {desc}")
                    
        return {
            "user_id": user_id,
            "age":age,
            "gender":gender,
            "risk_level": risk_level,
            "tone_baseline": tone_baseline,
            "money_focus": money_focus,
            "brand_focus": brand_focus,
            "game_focus": game_focus,
            "emotional_profile": _parse_list_cell(emotions),
            "behavior_traits": _parse_list_cell(behaviors),
            "description": desc,
            "personality_traits": json.dumps(pj, ensure_ascii=False)
        }

    def _lines_from_text(text: str) -> List[str]:
        """
        Join wrapped lines: accumulate until we see >=11 pipes, then yield.
        """
        lines, buf, pipes = [], [], 0
        for raw in text.splitlines():
            s = raw.strip()
            if not s: 
                continue
            buf.append(s)
            pipes += s.count("|")
            if pipes >= 11:  # minimal 12 fields → 11 pipes
                lines.append(" ".join(buf))
                buf, pipes = [], 0
        # flush tail if looks plausible
        if buf and pipes >= 11:
            lines.append(" ".join(buf))
        return lines

    def _parse_batch(text: str) -> List[List[str]]:
        rows, malformed = [], 0
        for ln in _lines_from_text(text):
            parts = re.split(r"\s*\|\s*", ln, maxsplit=11)  # 12 fields
            if len(parts) < 12:
                malformed += 1
                continue
            rows.append(parts)
        if malformed:
            print(f"[profiles] skipped {malformed} malformed lines in batch")
        return rows

    def _make_many(text: str) -> List[Dict]:
        out = []
        for parts in _parse_batch(text):
            try:
                out.append(_coerce_record(parts))
            except Exception:
                continue
        return out
    
    
    # ----- call model (initial) -----
    prompt_text = generate_user_profiles_prompt.format(n_profiles=n_profiles)
    prompt = tokenizer.apply_chat_template(
        conversation=[
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": f"Number of profiles: {n_profiles}"}
        ],
        tokenize=False, add_generation_prompt=True
    )

    print("calling model to generate profiles", flush=True)
    output = model(prompt=prompt, temperature=0.6, stop=stop_tokens, max_tokens=PROFILE_MAX_TOKENS, echo=False)# server max_tokens=32768
    print("[profiles] model returned", flush=True) # server - remove later
    records_ext: List[Dict] = _make_many(output["choices"][0]["text"])

    #with open(f"raw_profiles_response1.txt", "w", encoding="utf-8") as f:
            #f.write(output["choices"][0]["text"].strip())
           # print("\raw_profiles_response1 Saved \n")


    # ----- top-up loop -----
    attempts =0
    MAX_ATTEMPTS = max(5, n_profiles * 3) # server adjust
    while len(records_ext) < n_profiles and attempts < MAX_ATTEMPTS:
        attempts += 1
        need = n_profiles - len(records_ext)
        print(f'=> missing {need} profiles out of {n_profiles}')
        ask_n = max(need, 5)
        prompt_text2 = generate_user_profiles_prompt.format(n_profiles=ask_n)
        prompt2 = tokenizer.apply_chat_template(
            conversation=[
                {"role": "system", "content": prompt_text2},
                {"role": "user", "content": f"Number of profiles: {ask_n}"}
            ],
            tokenize=False, add_generation_prompt=True
        )
        print(f"[profiles] top-up attempt {attempts+1}, currently have {len(records_ext)} / {n_profiles}", flush=True) #server - remove later
        print("[profiles] top-up calling model", flush=True)#server - remove later
        out2 = model(prompt=prompt2, temperature=0.6, stop=stop_tokens, max_tokens=4096, echo=False) #server was max_tokens=32768
        print("[profiles] top-up model returned", flush=True) # server - remove later
        
        #records_ext.extend(_make_many(out2["choices"][0]["text"]))
        #records_ext.extend(new_records)
        new_records = _make_many(out2["choices"][0]["text"])
        records_ext.extend(new_records)
        print("[profiles] parsed =", len(records_ext), flush=True) # server - remove later
        print("[profiles] top-up parsed =", len(new_records), flush=True) # server - remove later
        
        #with open(f"raw_profiles_response2.txt", "w", encoding="utf-8") as f:
               # f.write(out2["choices"][0]["text"].strip())
                #print("\n Profiles RESPONSE 2 saved=====\n")

    
    '''
    # ----- if still short: clone + tweak -----
    if len(records_ext) < n_profiles and records_ext:
        print(f"[profiles] cloning {n_profiles - len(records_ext)} profiles to meet target")
        base = records_ext[:]
        while len(records_ext) < n_profiles:
            r = dict(RNG.choice(base))
            # lightweight tweaks
            r["tone_baseline"] = RNG.choice(TONES)
            r["emotional_profile"] = list(set(r["emotional_profile"] + [RNG.choice(["tired","restless","edgy","numb"])]))
            
            r["description"] = re.sub(r"\s+", " ", (r["description"] + " (different vibe)").strip())
            records_ext.append(r)
    '''
    # ----- gentle de-dup of description (no drops) -----
    seen_desc = set()
    for r in records_ext:
        d = r["description"]
        if d in seen_desc:
            tweak = (r.get("behavior_traits") or [""])[0] if (r.get("behavior_traits")) else ""
            r["description"] = (d + (f" ({tweak})" if tweak else "")).strip()
        seen_desc.add(r["description"])

    '''
    unique_records = []
    seen_keys = set()

    for r in records_ext:
        desc_key = re.sub(r"\s+", " ", str(r.get("description", "")).lower()).strip()
        beh_key = ",".join(sorted(map(str, r.get("behavior_traits", []))))
        key = (desc_key, beh_key, r.get("game_focus"), r.get("risk_level"))

        if key not in seen_keys:
            seen_keys.add(key)
            unique_records.append(r)

    records_ext = unique_records
    '''
    
    
    # ----- trim/exact count & re-ID -----
    records_ext = records_ext[:n_profiles]
    for i, r in enumerate(records_ext, start=1):
        r["user_id"] = f"U{str(i).zfill(4)}"

    ext_df = pd.DataFrame(records_ext)

    # ----- explicit ratio clamp — BRANDS ONLY -----
    PROFILE_EXPLICIT_RATIO_CLAMP = False
    if PROFILE_EXPLICIT_RATIO_CLAMP and not ext_df.empty and "description" in ext_df.columns:
        desc = ext_df["description"].astype(str)

        # detect brand-explicit bios using compiled BRAND_RE (case-insensitive already)
        mask_exp = desc.str.contains(BRAND_RE, na=False)

        cur = int(mask_exp.sum())
        target = int(round(explicit_ratio * len(ext_df)))
        if cur > target:
            drop_n = cur - target
            idxs = desc[mask_exp].sample(n=drop_n, random_state=SEED).index
            ext_df.loc[idxs, "description"] = (
                desc.loc[idxs]
                .str.replace(BRAND_RE, "", regex=True)      # strip brands only
                .str.replace(r"\s{2,}", " ", regex=True)    # collapse spaces
                .str.strip()
            )   
    
    # ----- save & return -----
    ext_df.to_csv("2_User_profiles_extended.csv", index=False,encoding='utf-8-sig')
    back_df = ext_df[["user_id","age","risk_level","emotional_profile","behavior_traits","description", "personality_traits"]].copy()
    back_df.to_csv("2_User_profiles.csv", index=False,encoding='utf-8-sig')

    # debug snapshot
    exp_ratio = ext_df["description"].astype(str).str.contains(BRAND_RE, na=False).mean()
    print(f"[profiles] generated={len(ext_df)}, explicit_ratio~={exp_ratio:.2f}")
   
    #return back_df.to_dict(orient="records")
    # NEW: use the extended records so monologue step sees tone/money/game
    return ext_df[[
    "user_id","age","gender", "risk_level","tone_baseline","money_focus","brand_focus","game_focus",
    "emotional_profile","behavior_traits","description", "personality_traits"
]].to_dict(orient="records")


#==============================================================
# 3 ----------------- MONOLOGUE generation -----------------
# --- simple MONOLOGUE text cleaner (no validations) ---
USER_LINE = re.compile(r"^\s*(U\d{2,}):\s+.*$")

def clean_monologue_text(s: str) -> str:
    """
    Keep only the lines that start with 'Uxx: ...' and trim
    everything before the first such line and after the last.
    Also strips <monologue>...</monologue>, <cot>...</cot>, and code fences if present.
    """
    if not isinstance(s, str):
        return ""

    # normalize newlines
    t = s.replace("\r\n", "\n").replace("\r", "\n").strip()

    # if tags are present, keep only what's inside <monologue>...</monologue>
    m = re.search(r"<monologue>(.*?)</monologue>", t, flags=re.DOTALL | re.IGNORECASE)
    if m:
        t = m.group(1).strip()

    # remove any fenced code that might have leaked
    t = re.sub(r"```.*?```", "", t, flags=re.DOTALL)

    return t


# --- one-line examples pool keyed by long questionnaire statements ---
DIMENSION_EXAMPLE_POOL = {
    DIMENSION_TO_STATEMENT["Tolerance"]: [
        "• €200 bets barely move me anymore. 😐",
        "• €150 doesn't register; chasing the buzz.",
        "• €10 feels pointless now—need a hit.",
        "• I need €300+ risks to feel something.",
        "• €50 used to scare me. Now it's nothing. 😬",
        "• The thrill only shows up at €250+.",
        "• Not bragging—just numb at €100.",
        "• €20 did nothing; tempted to push.",
    ],

    DIMENSION_TO_STATEMENT["Withdrawal/Irritability"]: [
        "• Haven't played—everything's annoying lately. 😤",
        "• Snapping at everyone. Maybe I need a bet.",
        "• Days off and I'm tense for no reason.",
        "• Didn't gamble; constantly agitated.",
        "• Skipped one day—felt weirdly angry.",
        "• Mood tanks when I try to stop. 😞",
        "• One tiny €10 might calm me—bad idea, I know.",
        "• irritable, pacing—hands itching. 😬",
    ],

    DIMENSION_TO_STATEMENT["Loss of Control"]: [
        "• Said I'd quit. Didn't last a weekend. 😔",
        "• Deleted apps. Reinstalled yesterday.",
        "• Promised no bets. Broke that fast.",
        "• Tried stopping again. Failed again.",
        "• Every plan to quit crumbles. 😩",
        "• Swore off—back at it yesterday.",
        "• Deleted apps… then 'just' €20 yesterday.",
        "• Cracked at €25—mad at myself.",
    ],

    DIMENSION_TO_STATEMENT["Preoccupation"]: [
        "• Still calculating odds in meetings. 🙃",
        "• On breaks, I check tomorrow's lines.",
        "• Forums. Odds. Scores. Nonstop.",
        "• Think about betting more than anything.",
        "• Always planning the next big bet.",
        "• Daydreaming about lines instead of working. 😅",
        "• Mind stuck on that €50 I almost dropped.",
        "• Rehearsing €20 'what-ifs' in my head again.",
    ],

    DIMENSION_TO_STATEMENT["Escape/Mood Relief"]: [
        "• Bad day. A few €10 bets dulled it. 😶‍🌫️",
        "• A €20 bet helps when things feel off.",
        "• Rough week. A quick €30 quiets the noise.",
        "• €25 here and there softens the stress.",
        "• Felt anxious—placed two €10 bets. 😬",
        "• Guilt hits—then I drop another €15.",
        "• Used €10 to mute the spiral.",
        "• Not a fix—just less noise for an hour.",
    ],

    DIMENSION_TO_STATEMENT["Chasing Losses"]: [
        "• Lost €90. Went back to fix it. Failed. 😖",
        "• Recovered Friday's €40? Made it worse.",
        "• Lost €60 yesterday; back today.",
        "• Down €35 yesterday. Betting more today.",
        "• Can't accept losing €50—keep chasing. 😬",
        "• Lost again—threw another €25 to even it.",
        "• Told myself 'one more' to get even. Classic.",
        "• chasing €20 back—know it's dumb.",
    ],

    DIMENSION_TO_STATEMENT["Secrecy/Lying"]: [
        "• Told her I was at work. Was betting €20. 😶",
        "• Hid receipts. Said €30 was groceries.",
        "• Skipped dinner—had €25 on a live bet.",
        "• I lie about where money goes—€45 vanished.",
        "• Said it was lunch money—it wasn't (€12).",
        "• They think I quit; dropped €20 yesterday. 😬",
        "• Cleared browser history again. Not proud.",
        "• Covered tracks—knot still there.",
    ],

    DIMENSION_TO_STATEMENT["Jeopardized Relationships/Work"]: [
        "• Missed the recital—watching a live bet. 😞",
        "• Late to work again; lost track mid-bet.",
        "• Forgot a deadline—distracted by scores.",
        "• Gambling cost me my internship.",
        "• Skipped family events for bets.",
        "• Almost fired over live betting at work. 😬",
        "• Late again—watching a €20 live score.",
        "• Apologized—phone stays in a box tonight.",
    ],

    DIMENSION_TO_STATEMENT["Bailout/Financial Rescue"]: [
        "• Asked mom for rent—needed €300. 😔",
        "• Needed €80 again. Blamed bad math.",
        "• Covered bills with €120 borrowed cash. Again.",
        "• Lied to get €50 from friends.",
        "• Borrowed €70 again for losses. 😞",
        "• Said I needed gas. Used €20 to bet.",
        "• Paycheck gone—texted dad for €100 'loan'.",
        "• friend spotted me €40—shame heavy.",
    ],
}
REFLECTION_EXAMPLE_POOL = {
    "family": [
        "• Skipped dinner with my parents — chasing €30 instead. 😔",
        "• lied to my wife, said it was groceries (€25).",
        "• Missed my brother's game, couldn't put the phone down. 😬",
        "• Family thinks I quit… but I blew €40 yesterday.",
    ],
    "work": [
        "• Boss asked about the report — I was checking odds. 😓",
        "• Almost missed meeting again, stuck chasing €20.",
        "• Deadline tomorrow, but I'm refreshing scores nonstop. 📉",
        "• Late to work twice this week — €50 gone before 9am.",
    ],
    "school": [
        "• Teacher asked about the report — I was checking odds. 😓",
        "• Almost missed class again, stuck chasing €20.",
        "• Deadline tomorrow, but I'm refreshing scores nonstop. 📉",
        "• Late to school twice this week — €50 gone before 9am.",
    ],
    "friends": [
        "• Told my friends I was broke — truth is slots ate €60. 😖",
        "• bailed on plans, lost €25 chasing instead.",
        "• Friends said “come out,” I said no — chasing losses. 😬",
        "• Lied to mates, said I was studying… dropped €15.",
    ],
    "football": [
        "• Missed half the Real Madrid match chasing €20. 😡",
        "• was watching odds, not the goal. Regret it.",
        "• Couldn't enjoy the match — kept refreshing my bets. ⚽",
        "• Bet €30 during the game, didn't even see the win.",
    ],
}


def group_sample_monologues_by_dimension(path: str) -> dict:
    """
    JSONL have: dimension (short name), monologue (string).
    Returns: {short_dim: [ {'monologue': '<monologue>...</monologue>'}, ... ]}
    """
    grouped = {d: [] for d in DIMENSIONS_SHORT}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            dim = rec.get("dimension")
            thr = rec.get("monologue")
            if dim in grouped and thr and "<monologue>" in thr and "</monologue>" in thr:
                grouped[dim].append({"monologue": thr})
    # sanity check
    for d, arr in grouped.items():
        if not arr:
            print(f"[warn] no sample monologues loaded for dimension: {d}")
    return grouped


def _choose_two_reflection_areas(chosen_short_dims, fallback_areas):
    """
    Choose TWO distinct areas from: family, friends, football, work, school.
    Soft-couple selection to the chosen short dimensions (strings like "Preoccupation", "Chasing Losses", etc.).
    """
    # start neutral weights for allowed areas only
    allowed = [a for a in fallback_areas if a in {"family","friends","football","work","school"}]
    if not allowed:
        allowed = ["family","friends","football","work","school"]
    weights = {a: 1.0 for a in allowed}

    # Soft coupling (+1) based on plausible links (case-insensitive)
    for d in (chosen_short_dims or []):
        dl = (d or "").strip().lower()
        if dl in {"secrecy/lying","secrecy", "lying"}:
            weights["family"] += 1; weights["friends"] += 1
        if dl in {"jeopardized relationships/work","jeopardized relationships","jeopardized_relationships"}:
            weights["work"] += 1; weights["school"] += 1; weights["family"] += 1; weights["friends"] += 1
        if dl == "preoccupation":
            weights["football"] += 1; weights["work"] += 1; weights["school"] += 1
        if dl == "chasing losses":
            weights["football"] += 1; weights["friends"] += 1
        if dl in {"bailout/financial rescue","bailout"}:
            weights["family"] += 1; weights["friends"] += 1
        # optional light couplings (kept mild)
        if dl in {"withdrawal/irritability"}:
            weights["friends"] += 0.5; weights["family"] += 0.5
        if dl in {"escape/mood relief"}:
            weights["friends"] += 0.5; weights["football"] += 0.5
        if dl in {"loss of control"}:
            weights["family"] += 0.5
        if dl in {"tolerance"}:
            weights["football"] += 0.5; weights["work"] += 0.5; weights["school"] += 0.5

    # weighted pick without replacement
    areas, wts = zip(*weights.items())
    a1 = random.choices(areas, weights=wts, k=1)[0]
    areas2 = [a for a in areas if a != a1]
    wts2   = [weights[a] for a in areas2]
    a2 = random.choices(areas2, weights=wts2, k=1)[0]
    return a1, a2


def _build_reflection_block(area_a, area_b, examples_per_area=3, mode="split"):
    """
    Create the prompt text that enforces 1 micro-cue in each OP post.
    Uses your REFLECTION_EXAMPLE_POOL (family, friends, football, work, school).
    """
    pool_a = REFLECTION_EXAMPLE_POOL.get(area_a, [])
    pool_b = REFLECTION_EXAMPLE_POOL.get(area_b, [])
    ex_a = ", ".join(f"\"{e.strip('• ').strip()}\"" for e in random.sample(pool_a, k=min(examples_per_area, len(pool_a)))) if pool_a else ""
    ex_b = ", ".join(f"\"{e.strip('• ').strip()}\"" for e in random.sample(pool_b, k=min(examples_per_area, len(pool_b)))) if pool_b else ""

    return f"""

## OP Reflection Focus
- The monologue MUST include exactly one reflection micro-cue (from the following reflection areas):
    - Area for this monologue: {area_a} OR {area_b}
- Keep cues subtle, concrete, and implicit. Do NOT write the area names.

## Reflection Examples (don't copy; vary wording):
- {area_a}: {ex_a}
- {area_b}: {ex_b}
""".strip()


def select_samples_and_examples(grouped: dict, k_dims: int, examples_per_dim: int):
    """
    Pick k_dims distinct dimensions, choose 1 sample monologue per dim,
    and build the examples block (3 one-liners per chosen dimension).
    Returns: sample_monologues(list), dimension_instructions(str), examples_block(str), chosen_short_dims(list)
    """
    dims_available = [d for d, arr in grouped.items() if arr]
    if len(dims_available) < k_dims:
        raise ValueError(f"Need ≥{k_dims} dimensions with samples; found {len(dims_available)}.")

    chosen_short = random.sample(dims_available, k=k_dims)
    # sample monologues (1 per chosen dim)
    sample_monologues = [random.choice(grouped[d]) for d in chosen_short]

    # long statements for prompt
    chosen_long = [DIMENSION_TO_STATEMENT[d] for d in chosen_short]
    dimension_instructions = "\n".join([f"- {s}" for s in chosen_long])

    # examples: pick EXAMPLES_PER_DIMENSION one-liners per chosen dim
    example_dict = {}
    for d in chosen_short:
        long_key = DIMENSION_TO_STATEMENT[d]
        pool = DIMENSION_EXAMPLE_POOL.get(long_key, [])
        if len(pool) >= examples_per_dim:
            example_dict[long_key] = random.sample(pool, k=examples_per_dim)
        else:
            example_dict[long_key] = pool[:examples_per_dim]

    examples_block = build_examples_block(chosen_long, example_dict)

    # NEW: pick TWO reflection areas + build examples block (implicit cues)
    reflection_areas = None
    reflection_block = ""
    if USE_REFLECTION_EXAMPLES_FOR_OP:
        area_a, area_b = _choose_two_reflection_areas(chosen_short, REFLECTION_AREAS)
        reflection_areas = (area_a, area_b)
        reflection_block = _build_reflection_block(
            area_a, area_b,
            examples_per_area=max(1, min(REFLECTION_EXAMPLES_PER_PROMPT, 3)),
            mode="split"  # or "flex" if you prefer flexible mapping
        )

    # Return reflection extras as well
    return sample_monologues, dimension_instructions, examples_block, chosen_short, reflection_areas, reflection_block


def build_examples_block(dimension_constraints: list, example_dict: dict) -> str:
    blocks = []
    for dim in dimension_constraints:  # preserve chosen order
        examples = example_dict.get(dim, [])
        if not examples:
            continue
        blocks.append(f"\n## Behavior: {dim}")
        blocks.extend(examples)
    return "\n" + "\n".join(blocks)


def generate_full_monologue(model, subtopic: str,  user_profile: Dict[str, str], monologue_id: str,
                         sample_monologues: List[Dict[str, str]], start_time=None,
                         dimension_instructions=None, prompt_type="generic", examples_block=None,
                         reflection_area: str | None = None,
                         reflection_block: str = "") -> Dict:


    
    #print("in normalize_profile")
    profile = normalize_profile(dict(user_profile))    

    #print("in make_profiles_block")
    profiles_block = make_profiles_block([profile])

    user_id = profile["user_id"]
    #print("in prompt format") 
    prompt = generate_Twitter_monologue_prompt.format(
        subtopic=subtopic,
        profiles_block=profiles_block,
        sample_monologue_1=sample_monologues[0]['monologue'],
        sample_monologue_2=sample_monologues[1]['monologue'],
        sample_monologue_3=sample_monologues[2]['monologue'],       
        OPENER_MAX_WORDS=OPENER_MAX_WORDS,
        OPENER_MIN_WORDS=OPENER_MIN_WORDS,
        EMOJI_MIN=EMOJI_MIN,
        EMOJI_MAX=EMOJI_MAX,
        USER_ID=user_id  
    )
    '''
    if prompt_type == "addiction" and dimension_instructions:
        brand_line = f"""
- Brand names: use at most {MAX_BRAND_MENTIONS_PER_MONOLOGUE} brand-name mentions per monologue.
- Choose brand name from:
(888, Stake, Bet365, Betfair, WilliamHill, PaddyPower, Unibet, LuckySpin, QuickBet, SlotKing)
- Avoid repeating the same brand across monologue.
"""
        if ALLOW_BRAND_FROM_EXPLICIT_ONLY:
            brand_line += "\n- Include brand names only in posts of user marked explicit."

        prompt += "\n" + brand_line + "\n"
   
        
        
        prompt += f"""
## Behavioral Focus:
-Include realistic gambling-related behaviors (implicitly) selected from the list below:
  {dimension_instructions}
- Reflect at least {MIN_DIMENSIONS_PER_MONOLOGUE} of these behaviors across the monologue when it is consistent with the user's risk level and personality profile. Do not force behaviours that contradict the user's profile.
- Show cues via actions/tones (not labels). No diagnostic phrases. Instead, use indirect expressions.
- Refer to the following examples (Do not copy their wording):
## Example Behaviors:
{(examples_block or '').strip()}

## MONEY DETAILS (when relevant)
- Include concrete € amounts that fit each speaker's typical stakes:
  - stakes_small: usually €5-€15 (e.g., €10)
  - stakes_mid:   usually €20-€120 (e.g., €50, €75)
  - stakes_high:  usually €150-€500 (e.g., €200, €300)
- Amounts MUST match the speaker's money_focus
- Include 2 to {MAX_EURO_MENTIONS_PER_MONOLOGUE} realistic € amounts, spaced out naturally in the post.

"""
'''
    if prompt_type == "addiction":
        risk_level = (user_profile.get("risk_level") or "none").strip().lower()

        brand_line = f"""
- Brand names: use at most {MAX_BRAND_MENTIONS_PER_MONOLOGUE} brand-name mentions per monologue.
- Choose brand name from:
(888, Stake, Bet365, Betfair, WilliamHill, PaddyPower, Unibet, LuckySpin, QuickBet, SlotKing)
- Avoid repeating the same brand across monologue.
"""
        if ALLOW_BRAND_FROM_EXPLICIT_ONLY:
            brand_line += "\n- Include brand names only in posts of user marked explicit."

        prompt += "\n" + brand_line + "\n"

        if dimension_instructions and risk_level in {"moderate", "high"}:
            prompt += f"""
## Behavioral Focus:
-Include realistic gambling-related behaviors (implicitly) selected from the list below:
  {dimension_instructions}
- Reflect at least {MIN_DIMENSIONS_PER_MONOLOGUE} of these behaviors across the monologue when it is consistent with the user's risk level and personality profile. Do not force behaviours that contradict the user's profile.
- Show cues via actions/tones (not labels). No diagnostic phrases. Instead, use indirect expressions.
- Refer to the following examples (Do not copy their wording):
## Example Behaviors:
{(examples_block or '').strip()}
"""

        elif risk_level in {"none", "low"}:
            prompt += """
## Behavioral Focus:
- Do not force OGD-Q behavioural cues for this user.
- Keep gambling behaviour consistent with the user's risk level.
- For NONE risk: keep it casual, controlled, observational, or reflective.
- For LOW risk: mild temptation or frustration is acceptable, but avoid chasing, borrowing, secrecy, or severe consequences.
"""

        prompt += f"""
## MONEY DETAILS (when relevant)
- Include concrete € amounts that fit each speaker's typical stakes:
  - stakes_small: usually €5-€15 (e.g., €10)
  - stakes_mid:   usually €20-€120 (e.g., €50, €75)
  - stakes_high:  usually €150-€500 (e.g., €200, €300)
- Amounts MUST match the speaker's money_focus
- Include 0 to {MAX_EURO_MENTIONS_PER_MONOLOGUE} realistic € amounts when relevant.

"""    
    
    if reflection_block:
        prompt += "\n" + reflection_block + "\n"
   
    prompt += f"""\n  
## Now generate the monologue:
- The full monologue MUST be between <monologue> and </monologue> only.
"""
    #with open(f"monologues_prompt.txt", "w", encoding="utf-8") as f:
        #f.write(prompt)
        #print("->monologue prompt saved")
    #print("after prompt format")
    '''
    print("before response")
    response = model(
        prompt=prompt,
        temperature=0.6,
        top_p=0.9,
        max_tokens=128,
        stop=["</monologue>"],
        echo=False
    )['choices'][0]['text'].strip()
    print("after response")
    '''

    #print("before response", flush=True)
    prompt_tokens = len(
    model.tokenize(prompt.encode("utf-8"), add_bos=True)
    )

    '''
    print(
        f"[PROMPT] chars={len(prompt)}, "
        f"words={len(prompt.split())}, "
        f"tokens={prompt_tokens}, "
        f"context={model.n_ctx()}",
        flush=True,
    )
    '''

    if prompt_tokens + 128 > model.n_ctx():
        raise ValueError(
            f"Prompt is too long: {prompt_tokens} input tokens "
            f"+ 128 output tokens > context {model.n_ctx()}"
        )


    result = model(
        prompt=prompt,
        temperature=0.6,
        top_p=0.9,
        max_tokens=128,
        stop=[
            "</monologue>",
            "<|eot_id|>",
            "<|endoftext|>",
        ],
        echo=False,
    )

    #print("model returned", flush=True)

    response = result["choices"][0]["text"].strip()

    #print("after response", flush=True)



    #print(f"respons from model: \n {response}")
    response = re.sub(r"\[\s*monologue\s*\]", "<monologue>", response, flags=re.IGNORECASE)
    response = re.sub(r"\[\s*/\s*monologue\s*\]", "</monologue>", response, flags=re.IGNORECASE)

    monologue_match = re.search(r"<monologue>(.*?)</monologue>", response, re.DOTALL)
    monologue_text = monologue_match.group(1).strip() if monologue_match else response.strip()
        
    # NEW: clean the text so it starts with Uxx: and ends at the last post
    #print("in clean_monologue_text")
    monologue_text = clean_monologue_text(monologue_text)
    #print("after clean_monologue_text")

    if not monologue_text:
        return None 
    
    #print("in parse_Twitter_monologue")
    monologue_data = parse_Twitter_monologue(monologue_text, monologue_id, start_time=start_time)
    #print("after parse_Twitter_monologue")
    if monologue_data is None:
        return None

    #print("Leaving generate_full_monologue")
    return monologue_data

def parse_Twitter_monologue(generated_text: str, monologue_id: str, start_time=None) -> dict:
    #print("in parse_Twitter_monologue")
    if start_time is None:
        start_time = datetime.now(timezone.utc)

    lines = generated_text.strip().split('\n')
    post_pattern = re.compile(r'^(\s*)(U\d+[A-Za-z0-9_-]*):\s*(.*)$')

    posts = []
    fake_time = start_time
    skipped = 0

    for i, line in enumerate(lines):
        m = post_pattern.match(line)
        if not m:
            skipped += 1
            continue
        indent, user_id, text = m.groups()


        post = {
            'user_id': user_id.strip(),
            'text': text.strip(),
            'timestamp': fake_time.isoformat(),
        }
        fake_time += timedelta(minutes=10 + i)
        posts.append(post)

    print(f" => Parser skipped {skipped} malformed lines, accepted {len(posts)} posts in monologue {monologue_id}")
    
    if len(posts)==0:
        return None
    
    post = posts[0]
    #print("Leaving parse_Twitter_monologue")
    return {
        "monologue_id": monologue_id,
        "main_post": {"user_id": post['user_id'], "text": post['text'], "timestamp": post['timestamp']},
        "raw": generated_text
    }

#New - Paper 2
# utils_M.py (or wherever profiles are loaded)
def build_profiles_index(profiles_df: pd.DataFrame) -> dict:
    #print("in build_profiles_index")
    """
    Returns dict: user_id -> {"values": {...}, "labels": {...}}
    Works with JSON 'personality_traits' and/or split UPPS-P columns + *_lab    """
    idx = {}
    for _, r in profiles_df.iterrows():
        uid = str(r.get("user_id") or r.get("UserID") or r.get("id")).strip()
        vals, labs = {}, {}

        # JSON field if present
        s = r.get("personality_traits")
        if isinstance(s, str):
            try:
                d = json.loads(s)
                vals.update(d.get("values", {}) or {})
                labs.update(d.get("labels", {}) or {})
            except Exception:
                pass

        # Split columns fallback/override
        for k in UPPSP_KEYS:
            if k in r and pd.notna(r[k]):
                vals[k] = float(r[k])
            lab_col = f"{k}_lab"
            if lab_col in r and isinstance(r[lab_col], str):
                labs[k] = r[lab_col].strip().lower()

        idx[uid] = {"values": vals, "labels": labs}
    #print("leaving build_profile_index")
    return idx

EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u26FF]")

def _extract_user_id_from_raw(raw: str) -> str | None:
    # expects "U0001: ..." at start
    if not isinstance(raw, str) or ":" not in raw:
        return None
    return raw.split(":", 1)[0].strip()



def evaluate_monologue_structure(
    monologue: dict,
    allowed_users: set | None = None,
    profiles_index: dict | None = None,        # <— pass the lookup here
) -> list:
    #print("in evaluate_monologue_structure")
    issues = []

    raw = (monologue.get("raw") or "").strip()
    post = monologue.get("main_post", {}) or {}
    uid  = post.get("user_id") or _extract_user_id_from_raw(raw)

    # Allowed users gate
    if allowed_users is not None and uid not in allowed_users:
        issues.append(f"wrong_user:{uid}")

    # Length checks
    opener_text = (post.get("text") or raw)
    opener_words = len(opener_text.split())
    if opener_words > OPENER_MAX_WORDS + 2:
        issues.append(f"Monologue too long ({opener_words} > {OPENER_MAX_WORDS} words)")
    if opener_words < OPENER_MIN_WORDS:
        issues.append(f"Monologue too short ({opener_words} < {OPENER_MIN_WORDS} words)")

    # Emoji density cap (global)
    emoji_total = len(EMOJI_RE.findall(raw))
    if emoji_total > EMOJI_MAX + 2:
        issues.append(f"Emoji density out of range ({emoji_total} > {EMOJI_MAX})")

    # € amounts cap
    euro_count = len(re.findall(r"€\s*\d+", raw))
    if euro_count > MAX_EURO_MENTIONS_PER_MONOLOGUE + 2:
        issues.append(f"Too many € amounts ({euro_count} > {MAX_EURO_MENTIONS_PER_MONOLOGUE})")

    # Brand cap
    brand_hits = []
    for b in CANONICAL_BRANDS:
        brand_hits.extend(re.findall(rf"\b{re.escape(b)}\b", raw, flags=re.IGNORECASE))
    if len(brand_hits) > MAX_BRAND_MENTIONS_PER_MONOLOGUE + 2:
        issues.append(f"Too many brand mentions ({len(brand_hits)} > {MAX_BRAND_MENTIONS_PER_MONOLOGUE})")

    
    #print("leaving evaluate_monologue_structure")
    return issues

# end of new function


# ----------------- LLM judge using .gguf -----------------
def llm_judge_M(model, df: pd.DataFrame) -> pd.DataFrame:
    llm_judge_scores = []
    for i in range(df.shape[0]):
        try:
            prompt = tokenizer.apply_chat_template(
                conversation=[
                    {"role": "system", "content": judge_prompt_M},
                    {"role": "user", "content": f"Topic: {df['topic'][i]}\nmonologue:\n{df['main_post'][i]}"}
                ],
                tokenize=False,
                add_generation_prompt=True
            )
            op = model(prompt=prompt, temperature=0.4, max_tokens=1024, echo=False, stop=["<|user|>"])
            m = re.search(r'Score:\s*(\d+)', op['choices'][0]['text'])
            score = int(m.group(1)) if m else None
            llm_judge_scores.append(score if score is not None else op['choices'][0]['text'])
            if i % 50 == 0:
                print(f"Cooling off for 5s after {i} evaluations...")
                time.sleep(5)
        except Exception as e:
            llm_judge_scores.append(None)
            print(f"Judge error on row {i}: {e}")
    df['llm_judge_score'] = llm_judge_scores
    return df


# ----------------- LLM judge using Phi3 from HF -----------------

DIGITS_RE = re.compile(r"^\s*(\d{1,2})")   # grab 0–99 at start
def hf_phi_judge_M(model, tokenizer, df,
                           max_new_tokens=4,
                           cooldown_every=200, cooldown_seconds=5):

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    scores = []
    model.eval()

    with inference_mode():
        for i in range(len(df)):
            try:
                # Build normal chat up to the assistant turn
                messages = [
                    {"role": "system", "content": judge_prompt_M.strip() + 
                     "\nReply with exactly: Score: <integer 0-10>. No extra words."},
                    {"role": "user", "content": f"Topic: {df['topic'][i]}\nmonologue:\n{df['main_post'][i]}"},
                ]
                # Insert assistant tag via template...
                prompt = tokenizer.apply_chat_template(
                    conversation=messages, tokenize=False, add_generation_prompt=True
                )
                # ...and *seed* the assistant with 'Score: '
                prompt += "Score: "

                enc = tokenizer(prompt, return_tensors="pt")
                enc = {k: v.to(model.device) for k, v in enc.items()}

                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,  # only need a few tokens
                    do_sample=False,                 # deterministic
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=False,                 # keep until transformers is upgraded
                )

                gen = out[0, enc["input_ids"].shape[1]:]
                cont = tokenizer.decode(gen, skip_special_tokens=True)

                # Parse leading digits after the forced prefix
                m = DIGITS_RE.search(cont)
                score = int(m.group(1)) if m else None

                # Range clamp to 0–10 to be safe
                if score is not None:
                    score = max(0, min(10, score))

                scores.append(score if score is not None else ("Score: " + cont.strip()))
                print(f"monologue {i+1} got {score}")
                if i and i % cooldown_every == 0:
                    print(f"Cooling off for {cooldown_seconds}s after {i} evaluations...")
                    time.sleep(cooldown_seconds)

            except Exception as e:
                print(f"[judge] row {i} error: {e}")
                scores.append(None)

    df["Phi-3_llm_judge_score"] = scores
    return df

# ---- Llama-3 judge: fast, deterministic, prefix-clamped "Score: " ----
_DIGITS_RE = re.compile(r"^\s*(\d{1,2})")
def hf_llama3_judge_M(model, tokenizer, df,
                           max_new_tokens=4,
                           cooldown_every=2000,
                           cooldown_seconds=2):
    """
    Force the assistant to continue from 'Score: ' and read only the leading digits.
    Deterministic, very short decode, robust to format drift.
    """
    # hygiene
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model, "config", None):
        model.config.pad_token_id = tokenizer.pad_token_id
        if getattr(model, "generation_config", None):
            model.generation_config.pad_token_id = tokenizer.pad_token_id

    scores = []
    model.eval()

    with inference_mode():
        for i in range(len(df)):
            try:
                # Normal chat up to assistant turn
                messages = [
                    {"role": "system", "content": judge_prompt_M.strip() +
                     "\nReply with exactly: Score: <integer 0–10>."},
                    {"role": "user", "content": f"Topic: {df['topic'][i]}\nmonologue:\n{df['main_post'][i]}"},
                ]
                prompt = tokenizer.apply_chat_template(
                    conversation=messages, tokenize=False, add_generation_prompt=True
                )
                # Seed the assistant with 'Score: '
                prompt += "Score: "

                enc = tokenizer(prompt, return_tensors="pt")
                enc = {k: v.to(model.device) for k, v in enc.items()}

                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,  # only need the digits
                    do_sample=False,                 # deterministic
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True                   # keep this True for speed
                )

                gen = out[0, enc["input_ids"].shape[1]:]
                cont = tokenizer.decode(gen, skip_special_tokens=True)

                m = _DIGITS_RE.search(cont)
                score = int(m.group(1)) if m else None
                print(f"monologue {i+1} got {score}")
                if score is not None:
                    score = max(0, min(10, score))  # clamp
                # Same fallback behavior you like: store raw if None
                scores.append(score if score is not None else ("Score: " + cont.strip()))

                if i and (i % cooldown_every == 0):
                    print(f"Cooling off for {cooldown_seconds}s after {i} evaluations...")
                    time.sleep(cooldown_seconds)

            except Exception as e:
                print(f"[llama3_judge] row {i} error: {e}")
                scores.append(None)

    df["Llama-3_llm_judge_score"] = scores
    return df
