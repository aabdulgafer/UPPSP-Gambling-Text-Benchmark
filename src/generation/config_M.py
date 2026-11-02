from pathlib import Path

#new
# -------- Runtime safety --------
PROFILE_MODEL_PATH = "models/Mistral/model.gguf"
MONOLOGUE_MODEL_PATH = "models/Llama3/model.gguf"

PROFILE_N_CTX = 4096
PROFILE_N_BATCH = 64
PROFILE_N_UBATCH = 32
PROFILE_N_THREADS = 2

MONOLOGUE_N_CTX = 8192
MONOLOGUE_N_BATCH = 128
MONOLOGUE_N_UBATCH = 64
MONOLOGUE_N_THREADS = 4

GPU_COOLDOWN_FREQUENCY = 150      # generation attempts
GPU_COOLDOWN_SECONDS = 120        # 2 minutes

PROFILE_MAX_TOKENS = 16384
PROFILE_BATCH_SIZE = 3
PROFILE_MAX_ATTEMPTS = 5

SUBTOPICS_CSV_PATH = r"Path\1_Generated_Subtopics.csv"


SAMPLE_MONOLOGUES_PATH = "monologue_samples.jsonl" 
topics_file_path = Path(r"Path\topics_M.txt")
csv_path         = Path(r"Path\3_Generated_Monologues.csv")
cooloff_period =60
cooloff_frequency= 30
Num_Monologues_Per_Subtopic= 2
# -------- Canonical dimension names --------
DIMENSIONS_SHORT = [
    "Tolerance",
    "Withdrawal/Irritability",
    "Loss of Control",
    "Preoccupation",
    "Escape/Mood Relief",
    "Chasing Losses",
    "Secrecy/Lying",
    "Jeopardized Relationships/Work",
    "Bailout/Financial Rescue",
]

DIMENSION_TO_STATEMENT = {
    "Tolerance": "*Needs to gamble with increasing amounts of money in order to achieve the desired excitement.",
    "Withdrawal/Irritability": "*Is restless or irritable when attempting to cut down or stop gambling.",
    "Loss of Control": "*Has made repeated unsuccessful efforts to control, cut back, or stop gambling.",
    "Preoccupation": "*Is often preoccupied with gambling (e.g., having persistent thoughts of reliving past gambling experiences, handicapping or planning the next venture, thinking of ways to get money with which to gamble).",
    "Escape/Mood Relief": "*Often gambles when feeling distressed (e.g., helpless, guilty, anxious, depressed).",
    "Chasing Losses": "*After losing money gambling, often returns another day to get even ('chasing' one's losses).",
    "Secrecy/Lying": "*Lies to conceal the extent of involvement with gambling.",
    "Jeopardized Relationships/Work": "*Has jeopardized or lost a significant relationship, job, or educational or career opportunity because of gambling.",
    "Bailout/Financial Rescue": "*Relies on others to provide money to relieve desperate financial situations caused by gambling.",
}

# --- Reflection controls (for OP realism) ---
USE_REFLECTION_EXAMPLES_FOR_OP = True          # turn off to revert to current behavior
REFLECTION_EXAMPLES_PER_PROMPT = 2             # 2 short cues per MONOLOGUE
REFLECTION_AREAS = ["family", "work","school", "friends", "football"]
ENFORCE_REFLECTION_AREAS = True     # NOT USED: require explicit area tokens in OP lines



# Money/brand density
MAX_EURO_MENTIONS_PER_MONOLOGUE = 5    
MAX_BRAND_MENTIONS_PER_MONOLOGUE = 2
BRAND_MONOLOGUE_PROB = 0.30            # NOT USED ~30% of gambling MONOLOGUEs may include a brand
ALLOW_MONEY_FROM_ALL_USERS = True   # NOT USED 
ALLOW_BRAND_FROM_EXPLICIT_ONLY = False # was true - changed to test on 3_9_2025

# -------- Generation scale --------
NUM_SUBTOPICS_PER_TOPIC = 10
NUM_PROFILES = 2000
SEED = 42
MIN_USERS_PER_MONOLOGUE=25
MAX_USERS_PER_MONOLOGUE=50
# -------- Participation / structure --------
MAX_MONOLOGUES_PER_USER   = 50          # cap participation for diversity


# -------- Profiles / explicitness policy --------
EXPLICIT_PROFILE_RATIO = 0.25        # ~25% brand/money explicit users
ALLOW_EXPLICIT_ONLY_IF_EXPLICIT_USER = True

# -------- Behavioral dimensions targets --------
MIN_DIMENSIONS_PER_MONOLOGUE    = 1     # hard minimum
TARGET_DIMENSIONS_PER_MONOLOGUE = 2     # soft target (used in prompts/retries)
EXAMPLES_PER_DIMENSION       = 3     # examples included per selected dimension

EMOJI_MIN = 0
EMOJI_MAX = 5                       # <= 5 emojis per MONOLOGUE
OPENER_MAX_WORDS = 30                # OP first post <= 30 words
OPENER_MIN_WORDS = 10

# -------- MONOLOGUE generation control --------
MONOLOGUE_GEN_MAX_RETRIES = 3          # retry attempts per subtopic
SAMPLES_PER_PROMPT     = 3           # sample_MONOLOGUEs k

# -------- Money ranges (used by prompts + post-check) --------
# Interpreted as € thresholds; None means open-ended
STAKES_EUR = {  # NOT USED
    "stakes_small": (0, 20),         # < €20
    "stakes_mid":   (20, 120),       # €20–€120
    "stakes_high":  (120, None),     # > €120
}

# -------- Forbidden diagnostic phrasing (prompt + sanitizer) --------
FORBIDDEN_DIAGNOSTIC_TERMS = ( # NOT USED
    "addiction",
    "addicted",
    "problem gambling",
    "diagnosis",
    "disorder",
    "i can't stop",
    "i need help",
    "i have a gambling problem",
)

# -------- CSV / export defaults (emoji-safe) --------
CSV_ENCODING       = "utf-8-sig"
CSV_LINETERMINATOR = "\n"
CSV_QUOTE_ALL      = True            # use csv.QUOTE_ALL where applicable



########## NOT USED


# --- Reflection distribution (soft) ---
REFLECTION_TWO_CUE_RATE = 0.35         # ~35% MONOLOGUEs get 2 cues, others 1

# --- Money / brand realism ---
BRANDS_ALLOWED_IN_OP = False
MAX_BRAND_MONOLOGUES_RATIO = 0.15         # allow brand refs in replies only
TARGET_MONEY_MONOLOGUES_RATIO = 0.60      # aim lower than current 
MAX_OP_MONEY_MONOLOGUES_RATIO = 0.35      # keep OPs less money-centric

# --- Interaction (mentions) ---
MENTION_MIN_RATIO = 0.15               # ensure some direct mentions overall

# --- Dimension re-weight to recover weak signals ---
DIMENSION_BOOST = {
    "Escape/Mood Relief": 1.5,
    "Tolerance": 1.5,
}


METRIC_INFO = {
  "g_eval": {"min": 0.0, "max": 3.0, "higher_better": True},
  "llm_judge_score": {"min": 0.0, "max": 10.0, "higher_better": True}
}
