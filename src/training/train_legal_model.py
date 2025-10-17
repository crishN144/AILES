#!/usr/bin/env python3
"""
Direct XML Processing Script for AILES Legal AI (Improved Version)
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
    OUTPUT_FILE = "/users/bgxp240/ailes_legal_ai/data/processed/improved_xml_dataset.jsonl"
    MODEL_PATH = "/mnt/scratch/bgxp240/models/models--mistralai--Mistral-Nemo-Instruct-2407/snapshots/04d8a90549d23fc6bd7f642064003592df51e9b3/"
    
    # Optimized token limits for direct XML processing
    MAX_INPUT_LENGTH = 20000  # Allow longer input for full XML context
    MAX_NEW_TOKENS = 1500     # Increased for complex legal analysis
    XML_TRUNCATE_LENGTH = 25000  # Characters to keep from XML
    MAX_RETRIES = 2           # Retry failed generations
    
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

def generate_text_with_xml(model, tokenizer, prompt, max_new_tokens=1500, retry_count=0):
    """Generate text from XML with proper token management and retry logic"""
    try:
        # Tokenize with proper attention mask handling
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=Config.MAX_INPUT_LENGTH,
            padding=True,  # Enable padding
            return_attention_mask=True  # Generate attention mask
        )
        
        input_ids = inputs['input_ids'].to(Config.DEVICE)
        attention_mask = inputs['attention_mask'].to(Config.DEVICE)
        
        input_length = input_ids.shape[1]
        print(f"    Input tokens: {input_length}")
        
        # Adaptive token management
        if input_length > Config.MAX_INPUT_LENGTH * 0.95:  # Close to limit
            print(f"    Input close to token limit, reducing max_new_tokens")
            max_new_tokens = min(max_new_tokens, 800)
        elif input_length > Config.MAX_INPUT_LENGTH * 0.8:
            max_new_tokens = min(max_new_tokens, 1200)
        
        # Generate with careful token management
        with torch.no_grad():
            # Adjust temperature for structured extraction
            temp = 0.5 if "structured_extraction" in prompt else 0.3
            
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,  # Pass attention mask
                max_new_tokens=max_new_tokens,
                temperature=temp,  # Dynamic temperature
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,  # Reduce repetition
                no_repeat_ngram_size=3   # Prevent repetitive phrases
            )
        
        # Decode only the new tokens
        new_tokens = outputs[0][input_length:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Validate response quality
        response = response.strip()
        if len(response) < 50:  # Too short
            raise ValueError(f"Response too short: {len(response)} characters")
        
        print(f"    Generated tokens: {len(new_tokens)}")
        return response
        
    except Exception as e:
        print(f"    Generation error (attempt {retry_count + 1}): {e}")
        
        # Retry logic
        if retry_count < Config.MAX_RETRIES:
            print(f"    Retrying generation... (attempt {retry_count + 2})")
            time.sleep(1)  # Brief pause before retry
            return generate_text_with_xml(model, tokenizer, prompt, max_new_tokens, retry_count + 1)
        else:
            print(f"    Failed after {Config.MAX_RETRIES + 1} attempts")
            return f"Error: Failed to generate after {Config.MAX_RETRIES + 1} attempts: {str(e)}"

def create_training_pairs_from_xml(xml_content, file_name, model, tokenizer):
    """Create multiple training pairs from raw XML content with improved validation"""
    pairs = []
    successful_pairs = 0
    failed_pairs = []
    
    # Define improved pair configurations
    pair_configs = [
        {
            "type": "fact_extraction",
            "max_tokens": 600,
            "prompt_template": """Analyze this UK family law judgment XML document and extract the key facts:

{xml_content}

Task: Create a concise summary of the essential facts from this case, identifying any legal statutes, acts, or legal principles mentioned. Focus on:
- Parties involved and their relationships
- Key events and circumstances
- Legal issues at stake
- Relevant dates and procedures

Summary:""",
            "user_preview": "Extract and summarize the key facts from this family law case XML:\n\n{xml_preview}..."
        },
        {
            "type": "legal_analysis", 
            "max_tokens": 900,
            "prompt_template": """Analyze this UK family law judgment XML document and provide legal analysis:

{xml_content}

Task: Generate a legal analysis report explaining the court's reasoning and decision. Include:
- Legal framework and statutes applied
- Key evidence considered
- Judicial reasoning process
- Final conclusions and orders
- Precedents or legal principles cited

Legal Analysis:""",
            "user_preview": "Provide legal analysis and reasoning for this family law judgment:\n\n{xml_preview}..."
        },
        {
            "type": "structured_extraction",
            "max_tokens": 800,
            "prompt_template": """Extract key information from this UK family law judgment XML as JSON:

{xml_content}

Task: Output a JSON object with these keys (use 'Not specified' if unavailable):
{{"case_citation": "", "court": "", "judge": "", "parties": "", "legal_issues": "", "key_statutes": "", "decision_summary": "", "orders": ""}}

JSON Output:""",
            "user_preview": "Extract key structured information from this family law case:\n\n{xml_preview}..."
        }
    ]
    
    xml_preview = xml_content[:1000] if len(xml_content) > 1000 else xml_content
    
    for config in pair_configs:
        print(f"  Generating {config['type']} pair...")
        
        try:
            # Format the prompt
            prompt = config["prompt_template"].format(xml_content=xml_content)
            user_content = config["user_preview"].format(xml_preview=xml_preview)
            
            # Generate response with retry logic
            response = generate_text_with_xml(model, tokenizer, prompt, config["max_tokens"])
            
            # Validate response
            if response.startswith("Error:"):
                print(f"    ❌ Failed to generate {config['type']}: {response}")
                failed_pairs.append(config['type'])
                continue
                
            if len(response.strip()) < 80:  # Minimum viable response (reduced threshold)
                print(f"    ❌ Response too short for {config['type']}: {len(response)} chars")
                failed_pairs.append(config['type'])
                continue
            
            # Check for repetitive or low-quality content (relaxed threshold)
            if len(set(response.split())) < len(response.split()) * 0.4:  # Relaxed from 0.6 to 0.4
                print(f"    ❌ Response too repetitive for {config['type']}")
                failed_pairs.append(config['type'])
                continue
            
            # Create valid pair
            pair = {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": response}
                ],
                "metadata": {
                    "source_file": file_name,
                    "pair_type": config['type'],
                    "response_length": len(response),
                    "generation_success": True,
                    "unique_words": len(set(response.split()))
                }
            }
            
            pairs.append(pair)
            successful_pairs += 1
            print(f"    ✅ Generated {config['type']} ({len(response)} chars, {len(set(response.split()))} unique words)")
            
        except Exception as e:
            print(f"    ❌ Exception in {config['type']}: {str(e)}")
            failed_pairs.append(config['type'])
            continue
    
    # Summary for this file
    print(f"  File summary: {successful_pairs}/3 pairs successful")
    if failed_pairs:
        print(f"  Failed pair types: {', '.join(failed_pairs)}")
    
    return pairs, successful_pairs, failed_pairs

def main():
    """Main execution function with enhanced error handling and validation"""
    start_time = time.time()
    
    print(f"Starting Improved Direct XML Processing at {time.strftime('%H:%M:%S')}")
    print("="*60)
    
    # Select random files
    selected_files = select_random_xml_files(Config.INPUT_DIR, Config.NUM_TEST_FILES)
    
    if not selected_files:
        print("No XML files selected. Exiting.")
        return
    
    print(f"\nTarget: Process {len(selected_files)} randomly selected files")
    print(f"Goal: Generate 3 pairs per file = {len(selected_files) * 3} total pairs")
    
    # Setup model
    try:
        model, tokenizer = setup_model()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Create output directory
    os.makedirs(os.path.dirname(Config.OUTPUT_FILE), exist_ok=True)
    
    # Enhanced tracking
    total_pairs = 0
    successful_files = 0
    processing_stats = []
    overall_failed_pairs = []
    
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
            
            # Generate pairs with validation
            file_start_time = time.time()
            pairs, successful_pairs, failed_pairs = create_training_pairs_from_xml(
                xml_content, xml_file, model, tokenizer
            )
            file_processing_time = time.time() - file_start_time
            
            # Save pairs
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')
                total_pairs += 1
            
            # Track statistics
            file_stats = {
                'file': xml_file,
                'pairs_generated': len(pairs),
                'successful_pairs': successful_pairs,
                'failed_pairs': failed_pairs,
                'processing_time': file_processing_time,
                'xml_size': char_count,
                'success_rate': f"{successful_pairs}/3"
            }
            processing_stats.append(file_stats)
            overall_failed_pairs.extend(failed_pairs)
            
            if len(pairs) > 0:
                successful_files += 1
            
            print(f"File result: {successful_pairs}/3 pairs successful in {file_processing_time:.1f}s")
            
            # Time check
            elapsed = time.time() - start_time
            if elapsed > 5400:  # 1.5 hours
                print("\nTime limit reached, stopping...")
                break
    
    # Comprehensive final summary
    elapsed_time = time.time() - start_time
    print("\n" + "="*60)
    print("COMPREHENSIVE PROCESSING SUMMARY")
    print("="*60)
    
    # Basic stats
    target_pairs = len(selected_files) * 3
    success_rate = (total_pairs / target_pairs * 100) if target_pairs > 0 else 0
    
    print(f"Files processed: {successful_files}/{len(selected_files)}")
    print(f"Total pairs generated: {total_pairs}/{target_pairs} ({success_rate:.1f}%)")
    print(f"Average pairs per file: {total_pairs/len(selected_files):.1f}")
    print(f"Total time elapsed: {elapsed_time/60:.1f} minutes")
    print(f"Average time per file: {elapsed_time/len(selected_files)/60:.1f} minutes")
    
    # Pair type analysis
    from collections import Counter
    failed_pair_counts = Counter(overall_failed_pairs)
    if failed_pair_counts:
        print(f"\nFailed pair types:")
        for pair_type, count in failed_pair_counts.items():
            print(f"  {pair_type}: {count} failures")
    else:
        print(f"\nNo failed pairs - all pair types generated successfully!")
    
    # Detailed file-by-file stats
    print(f"\nDetailed Processing Stats:")
    for stat in processing_stats:
        status = "✅ COMPLETE" if stat['successful_pairs'] == 3 else f"⚠️  PARTIAL ({stat['success_rate']})"
        print(f"  {stat['file']}: {status}")
        print(f"    Time: {stat['processing_time']:.1f}s, Size: {stat['xml_size']:,} chars")
        if stat['failed_pairs']:
            print(f"    Failed: {', '.join(stat['failed_pairs'])}")
    
    print(f"\nOutput saved to: {Config.OUTPUT_FILE}")
    
    if total_pairs > 0:
        print(f"\nTo review output:")
        print(f"head -5 {Config.OUTPUT_FILE}")
        print(f"wc -l {Config.OUTPUT_FILE}")
        print(f"jq '.metadata.pair_type' {Config.OUTPUT_FILE} | sort | uniq -c")
        
        # Quality check commands
        print(f"\nQuality checks:")
        print(f"jq '.messages[1].content | length' {Config.OUTPUT_FILE} | sort -n")
        print(f"jq '.metadata.unique_words' {Config.OUTPUT_FILE} | sort -n")
    
    # Final assessment with improved thresholds
    if success_rate >= 85:
        print(f"\n🎉 EXCELLENT: {success_rate:.1f}% pair generation rate!")
        print("Ready for immediate scale-up to full 4,000 file dataset.")
    elif success_rate >= 70:
        print(f"\n✅ GOOD: {success_rate:.1f}% pair generation rate.")
        print("Ready for cautious scale-up. Consider testing on 50 files first.")
    elif success_rate >= 50:
        print(f"\n⚠️  FAIR: {success_rate:.1f}% pair generation rate.")
        print("Investigate failed pair types before scale-up.")
    else:
        print(f"\n❌ NEEDS WORK: {success_rate:.1f}% pair generation rate.")
        print("Review approach and fix major issues before proceeding.")

if __name__ == "__main__":
    main()