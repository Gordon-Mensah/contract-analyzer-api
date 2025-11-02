# core/config.py

CACHE_DIR = ".cache_disk"
CACHE_TTL = 60 * 60 * 24

STATE_FILE = ".session_state.json"
PERSONA_FILE = "personas.json"
FEEDBACK_FILE = "feedback_store.json"
WEIGHTS_FILE = "ranking_weights.json"
MODEL_FILE = "feedback_model.joblib"

DEFAULT_WEIGHTS = {"w_risk": 2.0, "w_sim": 1.0, "w_len": 0.5}
AUTO_TRAIN_THRESHOLD = 50
MIN_SAMPLES_FOR_MODEL = 20
MIN_POSITIVE_FOR_MODEL = 5
