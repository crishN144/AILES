#!/usr/bin/env python3
import json
import re
from pathlib import Path

def clean_training_data():
    data_dir = Path("data/processed")
    
    for component in ["chatbot", "predictor", "explainer"]:
        input_file = data_dir / f"{component}_training_data.jsonl"
        if not input_file.exists():
            continue
            
        cleaned_examples = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    example = json.loads(line.strip())
                    
                    # Clean unicode issues
                    for key in ['instruction', 'input', 'output']:
                        if key in example and isinstance(example[key], str):
                            text = example[key]
                            text = text.replace('\\u201c', '"').replace('\\u201d', '"')
                            text = text.replace('\\u2019', "'").replace('\\u2018', "'")
                            text = text.replace('\\u2013', '-').replace('\\u2014', '-')
                            text = re.sub(r'\\n', ' ', text)
                            text = re.sub(r'\s+', ' ', text).strip()
                            
                            # Limit very long content
                            if len(text) > 3000:
                                text = text[:3000] + "..."
                            
                            example[key] = text
                    
                    cleaned_examples.append(example)
                except:
                    continue
        
        # Write cleaned data
        with open(input_file, 'w', encoding='utf-8') as f:
            for example in cleaned_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        print(f"✅ Cleaned {len(cleaned_examples)} {component} examples")

if __name__ == "__main__":
    clean_training_data()
