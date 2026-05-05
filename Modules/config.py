"""
config.py
Tunable parameters and keyword taxonomy for the AI trend analysis pipeline.
"""

import os

DATA_DIR       = "data"
PAPERS_FILE    = os.path.join(DATA_DIR, "papers.csv")
TOPIC_FILE     = os.path.join(DATA_DIR, "topic_assignments.csv")
TREND_FILE     = os.path.join(DATA_DIR, "trend_data.json")
DRIFT_FILE     = os.path.join(DATA_DIR, "drift_events.json")
PROGRESS_FILE  = os.path.join(DATA_DIR, "fetch_progress.json")
DASHBOARD_FILE = "dashboard.html"

# --- Scraper ---
API_PAGE_SIZE      = 50
MAX_RETRIES        = 3
KEYWORDS_PER_BATCH = 3   # Keep low — arXiv returns HTTP 500 for long query URLs
RESULTS_PER_BATCH  = 100
YEAR_START         = 2019

# Topic modeling 
N_TOPICS      = 10
N_TOP_WORDS   = 8
MIN_DF        = 3
MAX_DF        = 0.80
MAX_FEATURES  = 5000
LDA_MAX_ITER  = 80
RANDOM_STATE  = 42

#  Trend analysis 
TIME_BIN          = "Q"     
MIN_PAPERS_IN_BIN = 5
DRIFT_THRESHOLD   = 0.005
TOP_N_KEYWORDS    = 12
MK_P_THRESHOLD    = 0.05

#  Keyword taxonomy 
KEYWORD_CATEGORIES = {
    "Large Language Models": [
        "large language model", "GPT", "BERT", "LLaMA", "instruction tuning",
        "RLHF", "chain of thought", "in-context learning", "prompt engineering",
        "retrieval augmented generation", "autoregressive language model",
        "constitutional AI", "ChatGPT", "fine-tuning language model",
    ],
    "Natural Language Processing": [
        "natural language processing", "text classification", "sentiment analysis",
        "machine translation", "named entity recognition", "question answering",
        "text summarization", "information extraction", "dialogue systems",
        "speech recognition", "word embeddings", "coreference resolution",
    ],
    "Computer Vision": [
        "image recognition", "object detection", "image segmentation",
        "convolutional neural network", "vision transformer", "ViT",
        "video understanding", "depth estimation", "3D reconstruction",
        "pose estimation", "super resolution", "image captioning",
    ],
    "Generative Models": [
        "diffusion model", "generative adversarial network",
        "variational autoencoder", "score-based generative model",
        "text-to-image synthesis", "image inpainting", "controllable generation",
        "flow matching", "consistency model",
    ],
    "Foundation and Multimodal Models": [
        "foundation model", "multimodal learning", "vision language model",
        "CLIP", "cross-modal learning", "visual question answering",
        "image-text alignment", "multimodal fusion",
    ],
    "Reinforcement Learning": [
        "reinforcement learning", "policy gradient", "Q-learning",
        "actor-critic", "multi-agent reinforcement learning",
        "reward shaping", "model-based reinforcement learning",
        "offline reinforcement learning", "safe reinforcement learning",
    ],
    "Graph Machine Learning": [
        "graph neural network", "graph attention network",
        "knowledge graph", "graph transformer", "link prediction",
        "node classification", "graph generation", "molecular graph learning",
    ],
    "Federated and Privacy ML": [
        "federated learning", "differential privacy",
        "privacy-preserving machine learning", "split learning",
        "secure aggregation",
    ],
    "Transfer and Continual Learning": [
        "transfer learning", "domain adaptation", "meta-learning",
        "few-shot learning", "zero-shot learning", "continual learning",
        "catastrophic forgetting", "curriculum learning",
    ],
    "AI Agents and Reasoning": [
        "AI agent", "autonomous agent", "code generation with LLM",
        "tool use in language models", "chain of thought reasoning",
        "program synthesis", "neural symbolic reasoning",
        "planning with language models",
    ],
    "AI Safety and Alignment": [
        "AI safety", "AI alignment", "model interpretability",
        "AI fairness", "bias mitigation", "adversarial robustness",
        "hallucination in language models", "trustworthy AI",
    ],
    "Efficient ML": [
        "model compression", "knowledge distillation", "network pruning",
        "quantization neural network", "efficient transformer",
        "mixture of experts", "state space model", "Mamba architecture",
        "neural architecture search", "parameter efficient fine-tuning", "LoRA",
    ],
}

AI_KEYWORDS = [kw for kws in KEYWORD_CATEGORIES.values() for kw in kws]
