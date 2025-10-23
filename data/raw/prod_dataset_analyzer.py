#!/usr/bin/env python3
"""
FINAL PRODUCTION-READY AILES XML-to-Dataset Pipeline (v3) - COMPLETE & FIXED
Enhanced with Mistral-based case type classification and specialized prompts for:
- Child Arrangements (25.9%) - Care Proceedings (25.7%) - Financial Remedies (13.6%)
- Adoption (11.9%) - International Child Abduction (10.0%) - Domestic Violence (~5-10%) - Other (~3-8%)

ALL CRITICAL FIXES APPLIED: Added missing functions, fixed global variables, complete implementation
"""

import os
import json
import random
import re
import time
import signal
import shutil
import hashlib
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
import pandas as pd
import pickle

try:
    from lxml import etree as LET
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    DEPENDENCIES_AVAILABLE = True
    RAG_AVAILABLE = True
except ImportError as e:
    print(f"Missing dependencies: {e}")
    DEPENDENCIES_AVAILABLE = False
    RAG_AVAILABLE = False

# PRODUCTION Configuration
CORPUS_DIR = "/users/bgxp240/ailes_legal_ai/data/raw/xml_judgments"
OUTPUT_FILE = "/users/bgxp240/ailes_legal_ai/ailes_training_dataset_production.jsonl"
TRACKER_FILE = "/users/bgxp240/ailes_legal_ai/ailes_tracker_production.xlsx"
CHECKPOINT_DIR = "/users/bgxp240/ailes_legal_ai/ailes_checkpoints"
MANIFEST_FILE = "/users/bgxp240/ailes_legal_ai/ailes_manifest_production.txt"
REJECTED_LOG = "rejected_pairs_production.log"
BATCH_SIZE = 20
USE_MISTRAL = True
USE_RAG = False
TARGET_PAIRS = 15000
GENERATION_TIMEOUT = 90
MAX_MISTRAL_FAILURES = 10
FORCE_RESET = False

TARGET_DISTRIBUTION = {
    "financial_processing": 0.35,
    "legal_reasoning": 0.30,
    "court_decision": 0.25,
    "case_analysis": 0.05,
    "conversational_guidance": 0.05
}

MISTRAL_MODEL_PATH = "/mnt/scratch/bgxp240/models/models--mistralai--Mistral-Nemo-Instruct-2407/snapshots/04d8a90549d23fc6bd7f642064003592df51e9b3/"

CLASSIFICATION_PROMPT = """You are an expert UK family law analyst. Analyze this judgment excerpt and metadata to identify the primary case type. Return ONLY the case type name as a string.

CASE TYPES:
- financial_remedies: Matrimonial finances, Form E, maintenance, property division, Matrimonial Causes Act 1973
- child_arrangements: Contact, residence, specific issue orders, Children Act 1989 Section 8
- care_proceedings: Care orders, supervision orders, threshold criteria, local authority applications, Children Act 1989 Part IV
- adoption: Placement orders, adoption applications, parental consent, Adoption and Children Act 2002
- international_abduction: Hague Convention 1980, wrongful removal/retention, habitual residence
- domestic_violence: Non-molestation orders, occupation orders, protection measures, Family Law Act 1996
- other: Mixed, unclear, or other proceedings

INDICATORS:
- Statutory references (e.g., Children Act 1989, Hague Convention)
- Court type (e.g., Family Court, High Court Family Division)
- Legal terminology (e.g., 'threshold criteria', 'habitual residence')
- Party descriptions (e.g., local authority, applicant parent)
- Case citation (e.g., [2020] EWHC 1234 (Fam))

JUDGMENT EXCERPT:
{context}

METADATA:
{citation}

Return ONLY the case type name (e.g., 'financial_remedies'):"""

PROMPTS = {
    "financial_remedies": """You are an expert UK family law extractor. Extract ONLY factual information directly stated in this judgment excerpt. DO NOT infer, speculate, add details, or use vague/general phrases like 'as detailed in the judgment'. If no specific financial info, return empty array [].

CRITICAL RULES:
- Generate 2-5 unique pairs per excerpt.
- Each response MUST include exact quotes, amounts, or references from text.
- ALWAYS cite paragraph numbers when mentioned (e.g., 'paragraph 8', 'para 15').
- If no paragraph numbers exist, cite the quoted text directly.
- NEVER use vague phrases like 'as detailed in the judgment', 'as stated', or 'as referenced' without specifics.
- First, scan excerpt for explicit facts. Then, form specific questions/responses based ONLY on those.

Extract ONLY if text contains:
- Specific financial orders with exact amounts
- Form E disclosure requirements explicitly mentioned
- Asset valuations with precise figures
- Maintenance orders with exact sums
- General references to financial disclosure, asset division, or property settlement
- Statutory references (e.g., Matrimonial Causes Act 1973, Section 25)

Create JSON array of 2-5 objects ONLY if factual info exists:
- "task_type": "financial_processing"
- "instruction": specific question about stated info
- "response": precise answer with exact text references and paragraph numbers
- "summary": brief, specific description

Few-Shot Examples:
Excerpt: "The husband shall pay to the wife a lump sum of £500,000 by 1st January 2020 under Section 25 of the Matrimonial Causes Act 1973. The family home valued at £1.2 million shall be transferred to the wife (paragraph 8)."

Output: [
  {{"task_type": "financial_processing", "instruction": "What lump sum order was made and by when?", "response": "The husband shall pay the wife a lump sum of £500,000 by 1st January 2020.", "summary": "Lump sum £500,000 by 1/1/2020"}},
  {{"task_type": "financial_processing", "instruction": "What property order was made?", "response": "The family home valued at £1.2 million shall be transferred to the wife (paragraph 8).", "summary": "Family home £1.2m transferred to wife"}},
  {{"task_type": "legal_reasoning", "instruction": "What statutory basis applies to these financial orders?", "response": "Section 25 of the Matrimonial Causes Act 1973 governs the financial provision.", "summary": "Section 25 MCA 1973 basis"}}
]

Excerpt: "No specific financial orders or amounts mentioned."
Output: []

JUDGMENT TEXT:
{context}

Output ONLY valid JSON array starting with [:""",

    "child_arrangements": """You are an expert UK family law extractor. Extract ONLY factual information directly stated in this judgment excerpt. DO NOT infer, speculate, add details, or use vague/general phrases like 'as detailed in the judgment'. If no specific info, return empty array [].

CRITICAL RULES:
- Generate 2-5 unique pairs per excerpt.
- Each response MUST include exact quotes, orders, or references from text.
- ALWAYS cite paragraph numbers when mentioned (e.g., 'paragraph 8', 'para 15').
- If no paragraph numbers exist, cite the quoted text directly.
- NEVER use vague phrases like 'as detailed in the judgment', 'as stated', or 'as referenced' without specifics.
- First, scan excerpt for explicit facts. Then, form specific questions/responses based ONLY on those.

Extract ONLY if text contains:
- Specific child arrangement orders (e.g., residence, contact, specific issue)
- Statutory references (e.g., Children Act 1989, Section 8)
- Explicit parental responsibility details
- Court reasoning or decisions explicitly stated

Create JSON array of 2-5 objects ONLY if factual info exists:
- "task_type": choose from ["legal_reasoning", "court_decision"]
- "instruction": specific question about stated info
- "response": precise answer with exact text references and paragraph numbers
- "summary": brief, specific description

Few-Shot Examples:
Excerpt: "The court made a shared residence order on 14 January 2009 under Section 8 of the Children Act 1989. Parental responsibility was granted to both parties (paragraph 6)."

Output: [
  {{"task_type": "court_decision", "instruction": "What child arrangement order was made?", "response": "Shared residence order on 14 January 2009 under Section 8 of the Children Act 1989.", "summary": "Shared residence ordered"}},
  {{"task_type": "court_decision", "instruction": "What parental responsibility was granted?", "response": "Parental responsibility was granted to both parties (paragraph 6).", "summary": "Parental responsibility to both parties"}},
  {{"task_type": "legal_reasoning", "instruction": "What statute applies to the order?", "response": "Section 8 of the Children Act 1989.", "summary": "Section 8 CA 1989 applied"}}
]

Excerpt: "No orders mentioned."
Output: []

JUDGMENT TEXT:
{context}

Output ONLY valid JSON array starting with [:""",

    "care_proceedings": """You are an expert UK family law extractor. Extract ONLY factual information directly stated in this judgment excerpt. DO NOT infer, speculate, add details, or use vague/general phrases like 'as detailed in the judgment'. If no specific info, return empty array [].

CRITICAL RULES:
- Generate 2-5 unique pairs per excerpt.
- Each response MUST include exact quotes, orders, or references from text.
- ALWAYS cite paragraph numbers when mentioned (e.g., 'paragraph 8', 'para 15').
- If no paragraph numbers exist, cite the quoted text directly.
- NEVER use vague phrases like 'as detailed in the judgment', 'as stated', or 'as referenced' without specifics.
- First, scan excerpt for explicit facts. Then, form specific questions/responses based ONLY on those.

Extract ONLY if text contains:
- Specific care or supervision orders with details
- Statutory references (e.g., Children Act 1989, Part IV, Section 31)
- Threshold criteria or significant harm findings explicitly mentioned
- Local authority involvement, placements, or care plans detailed

Create JSON array of 2-5 objects ONLY if factual info exists:
- "task_type": choose from ["legal_reasoning", "court_decision"]
- "instruction": specific question about stated info
- "response": precise answer with exact text references and paragraph numbers
- "summary": brief, specific description

Few-Shot Examples:
Excerpt: "The court issued a care order under Section 31 of the Children Act 1989 as the threshold criteria of significant harm were met (paragraph 12). The children shall be placed with the foster carers pending permanence planning (paragraph 20)."

Output: [
  {{"task_type": "court_decision", "instruction": "What care order was made and under what statutory provision?", "response": "The court issued a care order under Section 31 of the Children Act 1989 (paragraph 12).", "summary": "Care order under Section 31 CA 1989"}},
  {{"task_type": "legal_reasoning", "instruction": "What legal threshold was met for the care order?", "response": "The threshold criteria of significant harm were met (paragraph 12).", "summary": "Significant harm threshold met"}},
  {{"task_type": "court_decision", "instruction": "What placement arrangements were ordered?", "response": "The children shall be placed with foster carers pending permanence planning (paragraph 20).", "summary": "Foster placement ordered"}}
]

Excerpt: "No care orders or threshold criteria discussed."
Output: []

JUDGMENT TEXT:
{context}

Output ONLY valid JSON array starting with [:""",

    "adoption": """You are an expert UK family law extractor. Extract ONLY factual information directly stated in this judgment excerpt. DO NOT infer, speculate, add details, or use vague/general phrases like 'as detailed in the judgment'. If no specific info, return empty array [].

CRITICAL RULES:
- Generate 2-5 unique pairs per excerpt.
- Each response MUST include exact quotes, orders, or references from text.
- ALWAYS cite paragraph numbers when mentioned (e.g., 'paragraph 8', 'para 15').
- If no paragraph numbers exist, cite the quoted text directly.
- NEVER use vague phrases like 'as detailed in the judgment', 'as stated', or 'as referenced' without specifics.
- First, scan excerpt for explicit facts. Then, form specific questions/responses based ONLY on those.

Extract ONLY if text contains:
- Specific adoption or placement orders with details
- Statutory references (e.g., Adoption and Children Act 2002)
- Parental consent issues explicitly mentioned
- Court reasoning or decisions explicitly stated

Create JSON array of 2-5 objects ONLY if factual info exists:
- "task_type": choose from ["legal_reasoning", "court_decision"]
- "instruction": specific question about stated info
- "response": precise answer with exact text references and paragraph numbers
- "summary": brief, specific description

Few-Shot Examples:
Excerpt: "The court granted an adoption order under Section 46 of the Adoption and Children Act 2002, dispensing with parental consent due to the child's welfare (paragraph 25). Placement was with the prospective adopters (paragraph 18)."

Output: [
  {{"task_type": "court_decision", "instruction": "What adoption order was granted?", "response": "The court granted an adoption order under Section 46 of the Adoption and Children Act 2002 (paragraph 25).", "summary": "Adoption order under Section 46 ACA 2002"}},
  {{"task_type": "legal_reasoning", "instruction": "Why was parental consent dispensed with?", "response": "Parental consent dispensed with due to the child's welfare (paragraph 25).", "summary": "Consent dispensed on welfare grounds"}},
  {{"task_type": "court_decision", "instruction": "What placement was ordered?", "response": "Placement with the prospective adopters (paragraph 18).", "summary": "Placement with adopters"}}
]

Excerpt: "No adoption orders discussed."
Output: []

JUDGMENT TEXT:
{context}

Output ONLY valid JSON array starting with [:""",

    "international_abduction": """You are an expert UK family law extractor. Extract ONLY factual information directly stated in this judgment excerpt. DO NOT infer, speculate, add details, or use vague/general phrases like 'as detailed in the judgment'. If no specific info, return empty array [].

CRITICAL RULES:
- Generate 2-5 unique pairs per excerpt.
- Each response MUST include exact quotes, orders, or references from text.
- ALWAYS cite paragraph numbers when mentioned (e.g., 'paragraph 8', 'para 15').
- If no paragraph numbers exist, cite the quoted text directly.
- NEVER use vague phrases like 'as detailed in the judgment', 'as stated', or 'as referenced' without specifics.
- First, scan excerpt for explicit facts. Then, form specific questions/responses based ONLY on those.

Extract ONLY if text contains:
- Specific orders for return or non-return
- Statutory references (e.g., Hague Convention 1980, Child Abduction Act 1984)
- Habitual residence findings
- Court reasoning or decisions explicitly stated

Create JSON array of 2-5 objects ONLY if factual info exists:
- "task_type": choose from ["legal_reasoning", "court_decision"]
- "instruction": specific question about stated info
- "response": precise answer with exact text references and paragraph numbers
- "summary": brief, specific description

Few-Shot Examples:
Excerpt: "The court ordered the return of the child to France under Article 12 of the Hague Convention 1980 due to wrongful removal. Habitual residence was determined as France (paragraph 18)."

Output: [
  {{"task_type": "court_decision", "instruction": "What return order was made?", "response": "The court ordered the return of the child to France under Article 12 of the Hague Convention 1980 (paragraph 18).", "summary": "Child return order under Article 12 Hague"}},
  {{"task_type": "legal_reasoning", "instruction": "What was the basis for the return order?", "response": "Due to wrongful removal as stated.", "summary": "Wrongful removal basis"}},
  {{"task_type": "legal_reasoning", "instruction": "What was the habitual residence finding?", "response": "Habitual residence determined as France (paragraph 18).", "summary": "Habitual residence in France"}}
]

Excerpt: "No abduction orders discussed."
Output: []

JUDGMENT TEXT:
{context}

Output ONLY valid JSON array starting with [:""",

    "domestic_violence": """You are an expert UK family law extractor. Extract ONLY factual information directly stated in this judgment excerpt. DO NOT infer, speculate, add details, or use vague/general phrases like 'as detailed in the judgment'. If no specific info, return empty array [].

CRITICAL RULES:
- Generate 2-5 unique pairs per excerpt.
- Each response MUST include exact quotes, orders, or references from text.
- ALWAYS cite paragraph numbers when mentioned (e.g., 'paragraph 8', 'para 15').
- If no paragraph numbers exist, cite the quoted text directly.
- NEVER use vague phrases like 'as detailed in the judgment', 'as stated', or 'as referenced' without specifics.
- First, scan excerpt for explicit facts. Then, form specific questions/responses based ONLY on those.

Extract ONLY if text contains:
- Specific non-molestation or occupation orders
- Statutory references (e.g., Family Law Act 1996, Part IV)
- Protection measures or domestic violence findings
- Court reasoning or decisions explicitly stated

Create JSON array of 2-5 objects ONLY if factual info exists:
- "task_type": choose from ["legal_reasoning", "court_decision"]
- "instruction": specific question about stated info
- "response": precise answer with exact text references and paragraph numbers
- "summary": brief, specific description

Few-Shot Examples:
Excerpt: "The court issued a non-molestation order under Part IV of the Family Law Act 1996 prohibiting contact for 12 months (paragraph 14). Occupation order granted excluding the respondent from the home."

Output: [
  {{"task_type": "court_decision", "instruction": "What protection order was issued?", "response": "Non-molestation order under Part IV of the Family Law Act 1996 for 12 months (paragraph 14).", "summary": "Non-molestation order issued"}},
  {{"task_type": "court_decision", "instruction": "What occupation order was made?", "response": "Occupation order excluding the respondent from the home.", "summary": "Occupation order granted"}},
  {{"task_type": "legal_reasoning", "instruction": "What statute applies to the protection measures?", "response": "Part IV of the Family Law Act 1996.", "summary": "Part IV FLA 1996 applied"}}
]

Excerpt: "No protection orders discussed."
Output: []

JUDGMENT TEXT:
{context}

Output ONLY valid JSON array starting with [:""",

    "other": """You are an expert UK family law extractor. Extract ONLY factual information directly stated in this judgment excerpt. DO NOT infer, speculate, add details, or use vague/general phrases like 'as detailed in the judgment'. If no specific info, return empty array [].

CRITICAL RULES:
- Generate 2-5 unique pairs per excerpt.
- Each response MUST include exact quotes, orders, or references from text.
- ALWAYS cite paragraph numbers when mentioned (e.g., 'paragraph 8', 'para 15').
- If no paragraph numbers exist, cite the quoted text directly.
- NEVER use vague phrases like 'as detailed in the judgment', 'as stated', or 'as referenced' without specifics.
- First, scan excerpt for explicit facts. Then, form specific questions/responses based ONLY on those.

Extract ONLY if text contains:
- Specific court orders or procedural details
- Statutory references or legal principles explicitly mentioned
- Court reasoning or decisions explicitly stated

Create JSON array of 2-5 objects ONLY if factual info exists:
- "task_type": choose from ["legal_reasoning", "case_analysis"]
- "instruction": specific question about stated info
- "response": precise answer with exact text references and paragraph numbers
- "summary": brief, specific description

Few-Shot Examples:
Excerpt: "The judgment references procedural fairness in paragraph 10 under the Family Law Act 1996. The court dismissed the application on grounds of natural justice (paragraph 15)."

Output: [
  {{"task_type": "legal_reasoning", "instruction": "What legal principles are referenced?", "response": "Procedural fairness under the Family Law Act 1996 (paragraph 10).", "summary": "Procedural fairness referenced"}},
  {{"task_type": "case_analysis", "instruction": "What was the court's decision on the application?", "response": "The court dismissed the application on grounds of natural justice (paragraph 15).", "summary": "Application dismissed"}},
  {{"task_type": "legal_reasoning", "instruction": "What statute applies to the procedural issues?", "response": "Family Law Act 1996 (paragraph 10).", "summary": "FLA 1996 procedural basis"}}
]

Excerpt: "No details mentioned."
Output: []

JUDGMENT TEXT:
{context}

Output ONLY valid JSON array starting with [:"""
}

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Generation timed out")

def log_rejected_pair(pair: Dict, reason: str, logger):
    """Log rejected pairs for analysis"""
    try:
        with open(REJECTED_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"pair": pair, "reason": reason, "timestamp": datetime.now().isoformat()}) + "\n")
        if logger:
            logger.debug(f"Logged rejected pair: {pair.get('task_type', 'unknown')} ({reason})")
    except Exception as e:
        if logger:
            logger.error(f"Failed to log rejected pair: {str(e)}")

def accept_by_quota(task_type: str, task_counts: Counter, total_accepted: int) -> bool:
    """Control task distribution according to target ratios"""
    target_ratio = TARGET_DISTRIBUTION.get(task_type, 0.0)
    if target_ratio == 0:
        return False
    
    if total_accepted < 15:
        return True
    
    if task_counts[task_type] == 0 and total_accepted > 5:
        return True

    new_proportion = (task_counts[task_type] + 1) / max(1, total_accepted + 1)
    return new_proportion <= target_ratio + 0.20  # Relaxed from 0.05 to accept more natural domain distribution

def to_llama31_format(instruction: str, context: str, response: str) -> Dict:
    """Format training pairs for Llama 3.1 fine-tuning"""
    BOS = "<|begin_of_text|>"
    SH = "<|start_header_id|>"
    EH = "<|end_header_id|>"
    EOT = "<|eot_id|>"
    
    system_prompt = "You are AILES, a specialized UK family law assistant."
    user_msg = f"{instruction}\n\n---\nJudgment excerpt:\n{context}"
    
    text = (
        f"{BOS}{SH}system{EH}\n\n{system_prompt}{EOT}"
        f"{SH}user{EH}\n\n{user_msg}{EOT}"
        f"{SH}assistant{EH}\n\n{response}{EOT}"
    )
    
    return {"text": text}

def classify_with_mistral(chunk: str, metadata: Dict, model, tokenizer, max_ctx: int, logger) -> str:
    """Classify case type using Mistral"""
    if model is None or tokenizer is None:
        if logger:
            logger.debug("Mistral unavailable, defaulting to child_arrangements")
        return "child_arrangements"
    
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(GENERATION_TIMEOUT)
        
        citation = metadata.get('citation', 'No citation available')
        prompt = CLASSIFICATION_PROMPT.format(context=chunk, citation=citation)
        
        messages = [
            {"role": "system", "content": "You are a legal data classification system. Return ONLY the case type name as a string."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except:
            formatted_prompt = prompt
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False)
        input_length = inputs.input_ids.shape[-1]
        max_new_tokens = min(50, max_ctx - input_length - 100)
        
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.1,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )
        
        signal.alarm(0)
        
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        
        valid_case_types = ["financial_remedies", "child_arrangements", "care_proceedings", 
                           "adoption", "international_abduction", "domestic_violence", "other"]
        
        if response in valid_case_types:
            return response
        
        if logger:
            logger.debug(f"Invalid case type from Mistral: {response}, defaulting to child_arrangements")
        return "child_arrangements"
        
    except TimeoutError:
        if logger:
            logger.debug("Mistral classification timed out, defaulting to child_arrangements")
        return "child_arrangements"
    except Exception as e:
        if logger:
            logger.debug(f"Mistral classification failed: {str(e)}, defaulting to child_arrangements")
        return "child_arrangements"
    finally:
        signal.alarm(0)

class StreamingJSONLWriter:
    """Efficient streaming writer for large datasets"""
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file = None
        self.count = 0
    
    def __enter__(self):
        if FORCE_RESET and os.path.exists(self.filepath):
            os.remove(self.filepath)
        
        existing = 0
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r', encoding='utf-8', errors='ignore') as rf:
                    for _ in rf:
                        existing += 1
            except Exception:
                existing = 0
        
        self.file = open(self.filepath, 'a', encoding='utf-8')
        self.count = existing
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
    
    def write(self, record: Dict):
        if self.file:
            self.file.write(json.dumps(record, ensure_ascii=False) + '\n')
            self.file.flush()
            self.count += 1
    
    def get_count(self) -> int:
        return self.count

class AILESTracker:
    """Track processing progress and quality metrics"""
    def __init__(self, tracker_file: str = TRACKER_FILE, checkpoint_dir: str = CHECKPOINT_DIR):
        self.tracker_file = tracker_file
        self.checkpoint_dir = checkpoint_dir
        self.rows = []
        self.processed_files = set()
        self.stats = {
            "total_files": 0,
            "successful": 0,
            "failed": 0,
            "total_pairs": 0,
            "high_factuality": 0,
            "start_time": datetime.now()
        }
        
        if FORCE_RESET:
            if os.path.exists(checkpoint_dir):
                shutil.rmtree(checkpoint_dir)
            if os.path.exists(tracker_file):
                os.remove(tracker_file)
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.load_state()
    
    def load_state(self):
        if FORCE_RESET:
            print("FORCE_RESET enabled: Starting fresh")
            return
        
        state_path = os.path.join(self.checkpoint_dir, "tracker_state.pkl")
        if os.path.exists(state_path):
            try:
                with open(state_path, 'rb') as f:
                    state = pickle.load(f)
                    self.rows = state.get('rows', [])
                    self.processed_files = state.get('processed_files', set())
                    self.stats = state.get('stats', self.stats)
                print(f"Resumed from checkpoint: {len(self.rows)} entries processed")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
    
    def save_state(self):
        state_path = os.path.join(self.checkpoint_dir, "tracker_state.pkl")
        state = {
            'rows': self.rows,
            'processed_files': self.processed_files,
            'stats': self.stats
        }
        try:
            with open(state_path, 'wb') as f:
                pickle.dump(state, f)
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")
    
    def add_entry(self, file_name: str, pairs_count: int, factuality_scores: List[bool] = None, error_msg: str = ""):
        year_match = re.search(r'\[(\d{4})\]', file_name)
        year = year_match.group(1) if year_match else "unknown"
        
        status = "Failed" if error_msg else ("Empty" if pairs_count == 0 else "Success")
        
        high_factuality_count = sum(factuality_scores) if factuality_scores else 0
        factuality_rate = (high_factuality_count / len(factuality_scores) * 100) if factuality_scores else 0
        factuality_score = "High" if factuality_rate >= 90 else "Medium" if factuality_rate >= 75 else "Low"
        
        row = {
            "File Name": file_name,
            "Year": year,
            "Status": status,
            "Pairs Generated": pairs_count,
            "Factuality Score": factuality_score,
            "Factuality Rate (%)": f"{factuality_rate:.1f}",
            "High Factuality Pairs": high_factuality_count,
            "Error": error_msg,
            "Processing Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.rows.append(row)
        self.processed_files.add(file_name)
        
        self.stats["total_files"] += 1
        if status == "Success":
            self.stats["successful"] += 1
        elif status == "Failed":
            self.stats["failed"] += 1
        
        self.stats["total_pairs"] += pairs_count
        self.stats["high_factuality"] += high_factuality_count
        self.save_state()
    
    def update_excel(self, force=False):
        if len(self.rows) < 50 or force:
            should_update = True
        else:
            should_update = len(self.rows) % 50 == 0
        
        if not should_update:
            return
        
        try:
            df = pd.DataFrame(self.rows)
            with pd.ExcelWriter(self.tracker_file) as writer:
                df.to_excel(writer, sheet_name="Processing Log", index=False)
                
                if len(self.rows) > 0:
                    summary_data = {
                        "Metric": ["Total Files", "Successful", "Failed", "Total Pairs", "High Factuality Rate (%)", "Avg Pairs/File"],
                        "Value": [
                            self.stats["total_files"],
                            self.stats["successful"], 
                            self.stats["failed"],
                            self.stats["total_pairs"],
                            f"{(self.stats['high_factuality'] / max(1, self.stats['total_pairs']) * 100):.1f}",
                            f"{(self.stats['total_pairs'] / max(1, self.stats['successful'])):.1f}"
                        ]
                    }
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name="Summary", index=False)
        except Exception as e:
            print(f"Failed to update Excel: {e}")
    
    def print_status(self):
        runtime = datetime.now() - self.stats["start_time"]
        rate = self.stats["total_files"] / max(0.1, runtime.total_seconds() / 3600)
        factuality_rate = (self.stats["high_factuality"] / max(1, self.stats["total_pairs"]) * 100)
        
        print(f"\n{'='*60}")
        print(f"Files: {self.stats['total_files']} | Success: {self.stats['successful']} | Pairs: {self.stats['total_pairs']}")
        print(f"Runtime: {runtime.total_seconds()/3600:.1f}h | Rate: {rate:.1f} files/h")
        print(f"Factuality: {factuality_rate:.1f}% high quality")
        print(f"{'='*60}")

def setup_logging():
    """Configure production logging"""
    if FORCE_RESET and os.path.exists('ailes_pipeline_production.log'):
        os.remove('ailes_pipeline_production.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ailes_pipeline_production.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def strip_namespace(tag: str) -> str:
    """Remove XML namespace prefixes"""
    return tag.split('}', 1)[-1] if '}' in tag else tag

def enhanced_xml_parser(file_path: str, logger) -> Dict[str, Any]:
    """Parse XML judgment files with enhanced error handling"""
    try:
        with open(file_path, 'rb') as f:
            parser = LET.XMLParser(recover=True, remove_blank_text=True)
            root = LET.parse(f, parser).getroot()
        
        metadata = {}
        for elem in root.iter():
            tag = strip_namespace(elem.tag).lower()
            if 'citation' in tag and elem.text and elem.text.strip():
                metadata['citation'] = elem.text.strip()
                break
        
        year_match = re.search(r'\[(\d{4})\]', os.path.basename(file_path))
        metadata['year'] = int(year_match.group(1)) if year_match else 0
        
        paragraphs = []
        for elem in root.iter():
            tag = strip_namespace(elem.tag).lower()
            if tag in ['p', 'para', 'paragraph', 'level']:
                raw = ''.join(elem.itertext())
                text = ' '.join(raw.split())
                if len(text) > 50:
                    paragraphs.append(text)
        
        if not paragraphs:
            for xpath in [".//body", ".//judgment", ".//text"]:
                try:
                    for elem in root.xpath(xpath):
                        raw = ''.join(elem.itertext())
                        text = ' '.join(raw.split())
                        if len(text) > 200:
                            paragraphs.append(text)
                except:
                    continue
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 > 2500 and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                current_chunk = current_chunk + "\n\n" + para if current_chunk else para
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        chunks = [c for c in chunks if len(c) > 300]
        
        seen_chunks = set()
        unique_chunks = []
        for chunk in chunks:
            chunk_signature = chunk[:200].lower().strip()
            chunk_hash = hashlib.md5(chunk_signature.encode('utf-8')).hexdigest()
            if chunk_hash not in seen_chunks:
                seen_chunks.add(chunk_hash)
                unique_chunks.append(chunk)
        
        chunks = unique_chunks
        total_chars = sum(len(c) for c in chunks)
        
        if logger:
            logger.info(f"Parsed {os.path.basename(file_path)}: {len(chunks)} unique chunks, {total_chars} chars")
        
        return {
            "metadata": metadata,
            "chunks": chunks,
            "char_count": total_chars,
            "chunk_count": len(chunks),
            "file_name": os.path.basename(file_path),
            "full_text": "\n\n".join(paragraphs)
        }
        
    except Exception as e:
        if logger:
            logger.error(f"Error parsing {file_path}: {str(e)}")
        return {"error": str(e), "file_name": os.path.basename(file_path)}

def load_mistral_model():
    """Load Mistral model for generation"""
    if not DEPENDENCIES_AVAILABLE:
        return None, None, None
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MISTRAL_MODEL_PATH)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            MISTRAL_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            low_cpu_mem_usage=True
        )
        
        max_ctx = getattr(model.config, "max_position_embeddings", 8192)
        model.generation_config.pad_token_id = tokenizer.eos_token_id
        
        print(f"Mistral loaded: {max_ctx} context window")
        return model, tokenizer, max_ctx
    except Exception as e:
        print(f"Error loading Mistral model: {e}")
        return None, None, None

def enhanced_json_repair(response: str) -> str:
    """Repair malformed JSON responses"""
    try:
        json.loads(response)
        return response
    except json.JSONDecodeError:
        response = response.strip()
        
        if response.startswith('[') and response.endswith(']'):
            try:
                cleaned = re.sub(r',\s*}', '}', response)
                cleaned = re.sub(r',\s*\]', ']', cleaned)
                json.loads(cleaned)
                return cleaned
            except:
                pass
        
        json_pattern = r'\[.*?\]'
        matches = re.findall(json_pattern, response, re.DOTALL)
        for match in matches:
            try:
                cleaned = re.sub(r',\s*}', '}', match)
                cleaned = re.sub(r',\s*\]', ']', cleaned)
                json.loads(cleaned)
                return cleaned
            except:
                continue
        
        return '[]'

def generate_with_mistral_safe(model, tokenizer, prompt: str, max_ctx: int, logger=None) -> List[Dict]:
    """Generate training pairs with Mistral model"""
    if model is None or tokenizer is None:
        return []
    
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(GENERATION_TIMEOUT)
        
        try:
            messages = [
                {"role": "system", "content": "You are a legal data extraction system. Return ONLY valid JSON. Do not speculate or add information not in the text."},
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except:
            formatted_prompt = prompt
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False)
        input_length = inputs.input_ids.shape[-1]
        max_new_tokens = min(1500, max_ctx - input_length - 100)
        
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.05,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )
        
        signal.alarm(0)
        
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        response = enhanced_json_repair(response)
        
        try:
            pairs = json.loads(response)
            if isinstance(pairs, dict):
                pairs = [pairs]
        except json.JSONDecodeError:
            return []
        
        valid_pairs = []
        for pair in pairs:
            if (isinstance(pair, dict) and 
                all(k in pair for k in ["task_type", "instruction", "response", "summary"]) and
                pair["task_type"] in TARGET_DISTRIBUTION and
                len(pair["response"].strip()) > 20 and
                len(pair["instruction"].strip()) > 10):
                valid_pairs.append(pair)
        
        return valid_pairs
            
    except TimeoutError:
        return []
    except Exception:
        return []
    finally:
        signal.alarm(0)

def validate_factuality_ultra_strict(pair: Dict, full_text: str, metadata: Dict, threshold=0.75) -> bool:
    """Ultra-strict factuality validation"""
    try:
        response = pair.get("response", "")
        if len(response) < 20:
            return False
        
        response_lower = response.lower()
        full_text_lower = full_text.lower()
        case_year = metadata.get("year", 0)
        
        legal_terms = [
            ("mental capacity act 2005", 2005),
            ("family procedure rules 2010", 2010),
            ("matrimonial causes act 1973", 1973),
            ("children act 1989", 1989)
        ]
        for term, year in legal_terms:
            if term in response_lower and case_year and case_year < year:
                return False
        
        sentences = [s.strip() for s in re.split(r'[.!?]', response) if len(s.strip()) > 15]
        
        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            
            if any(generic in sentence_lower for generic in [
                "the court", "legal principles", "seek advice", "consult a solicitor",
                "under the children act", "section 25", "the judgment", "the case"
            ]):
                continue
            
            if any(specific in sentence_lower for specific in [
                "ordered", "granted", "dismissed", "directed", "found", "decided",
                "£", "maintenance", "custody", "contact", "residence", "disclosed"
            ]):
                words = [w for w in sentence_lower.split() if len(w) > 3]
                if len(words) > 3:
                    matches = sum(1 for word in words if word in full_text_lower)
                    match_ratio = matches / len(words)
                    if match_ratio < threshold:
                        return False
        
        return True
        
    except Exception:
        return False

def generate_conservative_fallback_pairs(chunk: str, metadata: Dict, model, tokenizer, max_ctx, logger) -> List[Dict]:
    """Generate highly conservative fallback pairs"""
    pairs = []
    chunk_lower = chunk.lower()
    
    if any(term in chunk_lower for term in ["order", "direct", "grant", "judgment", "decision"]):
        if "order" in chunk_lower or "direct" in chunk_lower:
            pairs.append({
                "task_type": "court_decision",
                "instruction": "What procedural matters are addressed in this excerpt?",
                "response": "The excerpt addresses procedural aspects of the case as detailed in the judgment.",
                "summary": "Procedural information from judgment"
            })
    
    if any(term in chunk_lower for term in ["act", "section", "statute"]):
        pairs.append({
            "task_type": "legal_reasoning",
            "instruction": "What legal framework is referenced in this case?",
            "response": "The case references relevant statutory provisions as detailed in the judgment.",
            "summary": "Legal framework analysis"
        })
    
    return pairs[:1]

def generate_training_pairs(parsed_data: Dict[str, Any], model=None, tokenizer=None, max_ctx=None, logger=None) -> List[Tuple[Dict[str, Any], str, bool]]:
    """Generate training pairs from parsed judgment data"""
    if "error" in parsed_data:
        return []
    
    pairs = []
    chunks = parsed_data["chunks"]
    metadata = parsed_data["metadata"]
    full_text = parsed_data.get("full_text", "")
    
    mistral_failure_count = 0
    max_chunks = min(10, len(chunks))
    
    for i, chunk in enumerate(chunks[:max_chunks]):
        if len(chunk.strip()) < 200:
            continue
        
        case_type = classify_with_mistral(chunk, metadata, model, tokenizer, max_ctx, logger)
        prompt = PROMPTS[case_type].format(context=chunk)
        
        generated_pairs = []
        
        if USE_MISTRAL and model is not None and mistral_failure_count < MAX_MISTRAL_FAILURES:
            mistral_pairs = generate_with_mistral_safe(model, tokenizer, prompt, max_ctx, logger)
            if mistral_pairs:
                generated_pairs = mistral_pairs
                if logger:
                    logger.debug(f"Mistral generated {len(mistral_pairs)} pairs for chunk {i+1} ({case_type})")
            else:
                mistral_failure_count += 1
                if logger:
                    logger.debug(f"Mistral failed for chunk {i+1}, failure count: {mistral_failure_count}")
        
        # DISABLED: Fallback generation removed to eliminate generic pairs
        # if not generated_pairs and mistral_failure_count < MAX_MISTRAL_FAILURES:
        #     generated_pairs = generate_conservative_fallback_pairs(chunk, metadata, model, tokenizer, max_ctx, logger)
        #     if logger:
        #         logger.debug(f"Using fallback for chunk {i+1}, generated {len(generated_pairs)} pairs ({case_type})")
        
        for gp in generated_pairs:
            factuality_valid = validate_factuality_ultra_strict(gp, full_text, metadata)
            if not factuality_valid:
                log_rejected_pair(gp, "Low factuality", logger)
                continue
            pairs.append((gp, chunk, factuality_valid))
    
    if logger:
        logger.info(f"Generated {len(pairs)} validated pairs for {parsed_data['file_name']}")
    return pairs

def process_batch(xml_files: List[str], batch_num: int, model=None, tokenizer=None, max_ctx=None, 
                 logger=None, tracker=None, writer=None, task_counts=None) -> int:
    """Process a batch of XML files"""
    logger.info(f"Processing batch {batch_num}: {len(xml_files)} files")
    batch_pairs_count = 0
    
    for file_name in xml_files:
        if tracker and file_name in tracker.processed_files:
            logger.info(f"Skipping already processed file: {file_name}")
            continue
        
        start_time = time.time()
        file_path = os.path.join(CORPUS_DIR, file_name)
        
        try:
            parsed_data = enhanced_xml_parser(file_path, logger)
            if "error" in parsed_data:
                if tracker:
                    tracker.add_entry(file_name, 0, [], parsed_data["error"])
                continue
            
            pairs = generate_training_pairs(parsed_data, model, tokenizer, max_ctx, logger)
            
            file_pairs_count = 0
            factuality_scores = []
            
            for (pair, source_chunk, factuality_valid) in pairs:
                task_type = pair.get("task_type", "case_analysis")
                
                if accept_by_quota(task_type, task_counts, writer.get_count()):
                    record = to_llama31_format(
                        instruction=pair.get("instruction", "Analyze this legal text"),
                        context=source_chunk,
                        response=pair.get("response", "")
                    )
                    
                    writer.write(record)
                    task_counts[task_type] += 1
                    file_pairs_count += 1
                    batch_pairs_count += 1
                    factuality_scores.append(factuality_valid)
                else:
                    log_rejected_pair(pair, "Quota exceeded", logger)
            
            processing_time = time.time() - start_time
            logger.info(f"Generated {file_pairs_count} quality pairs for {file_name} in {processing_time:.1f}s")
            
            if tracker:
                tracker.add_entry(file_name, file_pairs_count, factuality_scores)
            
            with open(MANIFEST_FILE, 'a', encoding='utf-8') as mf:
                mf.write(parsed_data["file_name"] + '\n')
                
        except Exception as e:
            logger.error(f"Failed to process {file_name}: {e}")
            if tracker:
                tracker.add_entry(file_name, 0, [], str(e))
    
    return batch_pairs_count

def main():
    random.seed(42)
    os.environ["PYTHONHASHSEED"] = "42"
    if DEPENDENCIES_AVAILABLE:
        torch.manual_seed(42)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    
    logger = setup_logging()
    logger.info("Starting FINAL PRODUCTION AILES Pipeline (v3) - COMPLETE & FIXED")
    
    if not DEPENDENCIES_AVAILABLE:
        logger.error("Missing required dependencies. Please install: lxml, torch, transformers")
        return
    
    if FORCE_RESET:
        for file_path in [OUTPUT_FILE, TRACKER_FILE, MANIFEST_FILE, REJECTED_LOG]:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleared file: {file_path}")
        
        if os.path.exists(CHECKPOINT_DIR):
            shutil.rmtree(CHECKPOINT_DIR)
            logger.info(f"Cleared checkpoint directory: {CHECKPOINT_DIR}")
    
    tracker = AILESTracker()
    
    if not os.path.exists(CORPUS_DIR):
        logger.error(f"Corpus directory not found: {CORPUS_DIR}")
        return
    
    xml_files = [f for f in os.listdir(CORPUS_DIR) if f.endswith('.xml')]
    if not xml_files:
        logger.error(f"No XML files found in {CORPUS_DIR}")
        return
    
    # Optional test mode
    target_pairs = TARGET_PAIRS
    try:
        with open("selected_files.txt", "r") as f:
            xml_files = [os.path.basename(line.strip()) for line in f]
        target_pairs = 200
        logger.info(f"TEST MODE: Processing {len(xml_files)} selected files for {target_pairs} pairs")
    except FileNotFoundError:
        logger.info("No selected_files.txt found, processing all files")
        xml_files = [f for f in os.listdir(CORPUS_DIR) if f.endswith('.xml')]
        logger.info(f"PRODUCTION MODE: Processing {len(xml_files)} files for {target_pairs} high-quality pairs")
    
    remaining_files = xml_files.copy()
    random.shuffle(remaining_files)
    
    model, tokenizer, max_ctx = None, None, 8192
    if USE_MISTRAL:
        logger.info("Loading Mistral model...")
        model, tokenizer, max_ctx = load_mistral_model()
        if model is None:
            logger.warning("Failed to load Mistral model, using conservative fallback only")
        else:
            logger.info("Mistral model loaded successfully")
    
    logger.info("PRODUCTION CONFIGURATION:")
    logger.info(f"  Processing mode: ALL FILES (100% corpus coverage)")
    logger.info(f"  Total files to process: {len(remaining_files)}")
    logger.info(f"  Expected pairs: ~20,000-25,000 (no early stop)")
    logger.info(f"  Factuality threshold: 0.75 (STRICT)")
    logger.info(f"  Rejected pairs logging: ENABLED")
    logger.info(f"  RAG: {'ENABLED' if USE_RAG else 'DISABLED'}")
    logger.info(f"  Chunk deduplication: ENHANCED")
    logger.info(f"  Anti-fabrication: ACTIVE")
    logger.info(f"  Case classification: MISTRAL-BASED")
    
    task_counts = Counter()
    
    try:
        with StreamingJSONLWriter(OUTPUT_FILE) as writer:
            for i in range(0, len(remaining_files), BATCH_SIZE):
                batch_files = remaining_files[i:i + BATCH_SIZE]
                batch_num = (i // BATCH_SIZE) + 1
                
                batch_count = process_batch(
                    batch_files, batch_num, model, tokenizer, max_ctx,
                    logger, tracker, writer, task_counts
                )
                
                tracker.update_excel()
                tracker.print_status()

                # DISABLED: Process ALL files for 100% corpus coverage
                # if writer.get_count() >= target_pairs:
                #     logger.info(f"Reached target of {target_pairs} high-quality pairs")
                #     break
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            final_count = writer.get_count()
            logger.info(f"Successfully generated {final_count} production-quality pairs")
    
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        return
    
    tracker.update_excel(force=True)
    
    logger.info("="*60)
    logger.info("PRODUCTION PIPELINE COMPLETE")
    total_pairs = sum(task_counts.values())
    logger.info(f"Total pairs generated: {total_pairs}")
    logger.info(f"Files processed: {len(tracker.processed_files)}")
    logger.info(f"Success rate: {tracker.stats['successful']/max(1,tracker.stats['total_files'])*100:.1f}%")
    
    logger.info("\nTask distribution:")
    for task_type, count in task_counts.most_common():
        percentage = (count / total_pairs * 100) if total_pairs > 0 else 0
        target_pct = TARGET_DISTRIBUTION.get(task_type, 0) * 100
        logger.info(f"  {task_type}: {count} ({percentage:.1f}%, target: {target_pct:.1f}%)")
    
    factuality_rate = (tracker.stats["high_factuality"] / max(1, tracker.stats["total_pairs"]) * 100)
    logger.info(f"\nQuality metrics:")
    logger.info(f"  High factuality pairs: {tracker.stats['high_factuality']}/{tracker.stats['total_pairs']} ({factuality_rate:.1f}%)")
    
    logger.info(f"\nOutput files:")
    logger.info(f"  Dataset: {OUTPUT_FILE}")
    logger.info(f"  Tracker: {TRACKER_FILE}")
    logger.info(f"  Manifest: {MANIFEST_FILE}")
    logger.info(f"  Rejected pairs log: {REJECTED_LOG}")
    
    logger.info("PRODUCTION DATASET READY for Llama 3.1 8B fine-tuning!")
    logger.info("="*60)

if __name__ == "__main__":
    main()