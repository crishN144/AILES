#!/usr/bin/env python3
"""
Direct XML Processing Script for AILES Legal AI
Feed raw XML directly to Mistral for training pair generation
"""

import os
import json
import time
import random
import glob
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
class Config:
    INPUT_DIR = "/users/bgxp240/ailes_legal_ai/data/raw/xml_judgments/"
    OUTPUT_FILE = "/users/bgxp240/ailes_legal_ai/data/processed/direct_xml_dataset.jsonl"
    MODEL_PATH = "/mnt/scratch/bgxp240/models/models--mistralai--Mistral-Nemo-Instruct-2407/snapshots/04d8a90549d23fc6bd7f642064003592df51e9b3/"
    
    # Optimized token limits for direct XML processing
    MAX_INPUT_LENGTH = 20000  # Allow longer input for full XML context
    MAX_NEW_TOKENS = 1000     # Allow fuller responses
    XML_TRUNCATE_LENGTH = 25000  # Characters to keep from XML
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_TEST_FILES = 5

def select_random_xml_files(input_dir, num_files=5):
    """Select random XML files from the input directory"""
    try:
        xml_pattern = os.path.join(input_dir, "*.xml")
        all_xml_files = glob.glob(xml_pattern)
        
        if not all_xml_files:
            print(f"No XML files found in {input_dir}")
            return []
        
        xml_filenames = [os.path.basename(f) for f in all_xml_files]
        print(f"Found {len(xml_filenames)} XML files in directory")
        
        if len(xml_filenames) <= num_files:
            selected_files = xml_filenames
            print(f"Using all {len(selected_files)} available files")
        else:
            selected_files = random.sample(xml_filenames, num_files)
            print(f"Randomly selected {num_files} files from {len(xml_filenames)} available")
        
        selected_files.sort()
        
        print("Selected files:")
        for i, filename in enumerate(selected_files, 1):
            print(f"  {i}. {filename}")
        
        return selected_files
        
    except Exception as e:
        print(f"Error selecting XML files: {e}")
        return []

def setup_model():
    """Load Mistral model with optimized settings for XML processing"""
    print("Loading Mistral model for direct XML processing...")
    print(f"Using device: {Config.DEVICE}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_PATH)
        
        # Ensure we have a pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_PATH,
            torch_dtype=torch.float16 if Config.DEVICE == "cuda" else torch.float32,
            device_map="auto" if Config.DEVICE == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        if Config.DEVICE == "cpu":
            model = model.to(Config.DEVICE)
        
        print(f"Model loaded successfully on {Config.DEVICE}")
        print(f"Model max length: {getattr(model.config, 'max_position_embeddings', 'Unknown')}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def read_and_preprocess_xml(xml_file_path):
    """Read XML file and preprocess for optimal token usage"""
    try:
        with open(xml_file_path, 'r', encoding='utf-8') as f:
            xml_content = f.read()
        
        # Remove excessive whitespace while preserving structure
        xml_content = '\n'.join(line.strip() for line in xml_content.split('\n') if line.strip())
        
        # Truncate if too long, keeping the beginning and end
        if len(xml_content) > Config.XML_TRUNCATE_LENGTH:
            # Keep first 20k chars and last 5k chars with indicator
            first_part = xml_content[:20000]
            last_part = xml_content[-5000:]
            xml_content = first_part + "\n\n[... middle content truncated ...]\n\n" + last_part
            print(f"XML truncated to {len(xml_content)} characters")
        
        return xml_content, len(xml_content)
        
    except Exception as e:
        print(f"Error reading XML file {xml_file_path}: {e}")
        return None, 0

def generate_text_with_xml(model, tokenizer, prompt, max_new_tokens=1000):
    """Generate text from XML with proper token management"""
    try:
        # Tokenize and check length
        inputs = tokenizer.encode(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=Config.MAX_INPUT_LENGTH
        )
        
        input_length = inputs.shape[1]
        print(f"Input tokens: {input_length}")
        
        if input_length > Config.MAX_INPUT_LENGTH * 0.95:  # Close to limit
            print(f"Input close to token limit, reducing max_new_tokens")
            max_new_tokens = min(max_new_tokens, 500)
        
        inputs = inputs.to(Config.DEVICE)
        
        # Generate with careful token management
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,  # Lower temperature for more focused legal analysis
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1  # Reduce repetition
            )
        
        # Decode only the new tokens
        new_tokens = outputs[0][input_length:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        print(f"Generated tokens: {len(new_tokens)}")
        return response.strip()
        
    except Exception as e:
        print(f"Generation error: {e}")
        return f"Error: {str(e)}"

def create_training_pairs_from_xml(xml_content, file_name, model, tokenizer):
    """Create multiple training pairs from raw XML content"""
    pairs = []
    
    # Pair 1: Fact Extraction and Summarization
    fact_prompt = f"""Analyze this UK family law judgment XML document and extract the key facts:

{xml_content}

Task: Create a concise summary of the essential facts from this case, identifying any legal statutes, acts, or legal principles mentioned. Focus on:
- Parties involved and their relationships
- Key events and circumstances
- Legal issues at stake
- Relevant dates and procedures

Summary:"""

    fact_response = generate_text_with_xml(model, tokenizer, fact_prompt, 400)
    
    if "Error:" not in fact_response:
        pairs.append({
            "messages": [
                {"role": "user", "content": f"Extract and summarize the key facts from this family law case XML:\n\n{xml_content[:1000]}..."},
                {"role": "assistant", "content": fact_response}
            ],
            "metadata": {
                "source_file": file_name,
                "pair_type": "fact_extraction"
            }
        })

    # Pair 2: Legal Analysis and Reasoning
    analysis_prompt = f"""Analyze this UK family law judgment XML document and provide legal analysis:

{xml_content}

Task: Generate a legal analysis report explaining the court's reasoning and decision. Include:
- Legal framework and statutes applied
- Key evidence considered
- Judicial reasoning process
- Final conclusions and orders
- Precedents or legal principles cited

Legal Analysis:"""

    analysis_response = generate_text_with_xml(model, tokenizer, analysis_prompt, 600)
    
    if "Error:" not in analysis_response:
        pairs.append({
            "messages": [
                {"role": "user", "content": f"Provide legal analysis and reasoning for this family law judgment:\n\n{xml_content[:1000]}..."},
                {"role": "assistant", "content": analysis_response}
            ],
            "metadata": {
                "source_file": file_name,
                "pair_type": "legal_analysis"
            }
        })

    # Pair 3: Structured Information Extraction
    structured_prompt = f"""Extract structured information from this UK family law judgment XML:

{xml_content}

Task: Create a structured summary in the following format:
- Case Citation: 
- Court:
- Judge:
- Parties:
- Legal Issues:
- Key Statutes/Acts:
- Decision Summary:
- Next Steps/Orders:

Structured Summary:"""

    structured_response = generate_text_with_xml(model, tokenizer, structured_prompt, 400)
    
    if "Error:" not in structured_response:
        pairs.append({
            "messages": [
                {"role": "user", "content": f"Extract structured information from this family law case:\n\n{xml_content[:1000]}..."},
                {"role": "assistant", "content": structured_response}
            ],
            "metadata": {
                "source_file": file_name,
                "pair_type": "structured_extraction"
            }
        })

    return pairs

def main():
    """Main execution function"""
    start_time = time.time()
    
    print(f"Starting Direct XML Processing at {time.strftime('%H:%M:%S')}")
    print("="*60)
    
    # Select random files
    selected_files = select_random_xml_files(Config.INPUT_DIR, Config.NUM_TEST_FILES)
    
    if not selected_files:
        print("No XML files selected. Exiting.")
        return
    
    print(f"\nTarget: Process {len(selected_files)} randomly selected files")
    
    # Setup model
    try:
        model, tokenizer = setup_model()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Create output directory
    os.makedirs(os.path.dirname(Config.OUTPUT_FILE), exist_ok=True)
    
    total_pairs = 0
    successful_files = 0
    processing_stats = []
    
    # Process each file
    with open(Config.OUTPUT_FILE, 'w') as f:
        for i, xml_file in enumerate(selected_files, 1):
            file_path = os.path.join(Config.INPUT_DIR, xml_file)
            
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue
            
            print(f"\n[{i}/{len(selected_files)}] Processing: {xml_file}")
            print("-" * 50)
            
            # Read XML
            xml_content, char_count = read_and_preprocess_xml(file_path)
            if not xml_content:
                print(f"Failed to read {xml_file}")
                continue
            
            print(f"XML size: {char_count:,} characters")
            
            # Generate pairs
            file_start_time = time.time()
            pairs = create_training_pairs_from_xml(xml_content, xml_file, model, tokenizer)
            file_processing_time = time.time() - file_start_time
            
            # Save pairs
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')
                total_pairs += 1
            
            successful_files += 1
            processing_stats.append({
                'file': xml_file,
                'pairs_generated': len(pairs),
                'processing_time': file_processing_time,
                'xml_size': char_count
            })
            
            print(f"Generated {len(pairs)} pairs in {file_processing_time:.1f}s")
            
            # Time check
            elapsed = time.time() - start_time
            if elapsed > 5400:  # 1.5 hours
                print("\nTime limit reached, stopping...")
                break
    
    # Final summary
    elapsed_time = time.time() - start_time
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Files processed: {successful_files}/{len(selected_files)}")
    print(f"Total pairs generated: {total_pairs}")
    print(f"Average pairs per file: {total_pairs/successful_files:.1f}")
    print(f"Total time elapsed: {elapsed_time/60:.1f} minutes")
    print(f"Average time per file: {elapsed_time/successful_files/60:.1f} minutes")
    print(f"Output saved to: {Config.OUTPUT_FILE}")
    
    # Detailed stats
    print(f"\nDetailed Processing Stats:")
    for stat in processing_stats:
        print(f"  {stat['file']}: {stat['pairs_generated']} pairs, "
              f"{stat['processing_time']:.1f}s, {stat['xml_size']:,} chars")
    
    if total_pairs > 0:
        print(f"\nTo review output:")
        print(f"head -5 {Config.OUTPUT_FILE}")
        print(f"wc -l {Config.OUTPUT_FILE}")
        print(f"jq '.metadata.pair_type' {Config.OUTPUT_FILE} | sort | uniq -c")

if __name__ == "__main__":
    main()