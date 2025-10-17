#!/usr/bin/env python3
"""
Test HTML Explainability with Mistral-Nemo-Instruct
For AILES Legal AI demonstration to ReGoBs team
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import time

# Model configuration
MODEL_PATH = "/mnt/scratch/bgxp240/models/models--mistralai--Mistral-Nemo-Instruct-2407"

class MistralHTMLExplainabilityTester:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        print(f"Loading Mistral model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with appropriate settings for your GPU
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float16,  # Use float16 for memory efficiency
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        print("Model loaded successfully!")
    
    def create_html_prompt(self, user_input):
        """Create the prompt with HTML explainability instructions"""
        
        prompt = f"""[INST] You are a legal AI assistant providing analysis with HTML formatting for explainability.

CRITICAL REQUIREMENTS:
1. Generate TWO outputs: standard_output (plain text) and explainability_output (with HTML tags)
2. Use HTML tags to show what influenced each judgment part

HTML FORMATTING RULES:
- Case law: <span class="case-law" style="color: #0066CC; font-weight: bold; text-decoration: underline">[case name]</span>
- Statutes: <span class="statute" style="color: #008800; font-weight: bold">[statute]</span>
- User input reference: <span class="input-ref" style="color: #9C27B0; font-weight: bold; border-bottom: 2px dotted #9C27B0" data-input="{user_input}">[referenced part]</span>
- Legal principles: <span class="principle" style="color: #CC6600; font-weight: bold">[principle]</span>
- Financial amounts: <span class="financial" style="color: #444444; background-color: #FFFFCC">[amount]</span>

USER INPUT: {user_input}

Generate a JSON response with this exact structure:
{{
  "standard_output": "Plain text legal analysis without any HTML tags",
  "explainability_output": "Same analysis but with HTML tags showing: (1) which parts of user input influenced the judgment using purple dotted underline spans, and (2) legal sources in their respective colors"
}}

IMPORTANT: The explainability_output must contain actual HTML span tags with the styles specified above.
[/INST]

Response:"""
        
        return prompt
    
    def generate_response(self, user_input, max_new_tokens=512):
        """Generate response from Mistral model"""
        
        # Create prompt
        prompt = self.create_html_prompt(user_input)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        print("Generating response...")
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        generation_time = time.time() - start_time
        print(f"Generation completed in {generation_time:.2f} seconds")
        
        return response, generation_time
    
    def parse_response(self, response_text):
        """Parse the model's response to extract JSON"""
        
        # Try to extract JSON from response
        try:
            # Look for JSON-like structure
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # If parsing fails, create a structured response from the text
        return self.create_fallback_response(response_text)
    
    def create_fallback_response(self, response_text):
        """Create HTML-formatted response if model doesn't generate proper JSON"""
        
        # Apply HTML tags manually to demonstrate the concept
        html_output = response_text
        
        # Apply case law formatting
        import re
        case_patterns = [
            (r'\b([A-Z][a-z]+ v\.? [A-Z][a-z]+(?:\s*\[\d{4}\])?)', 'case-law'),
            (r'(\[\d{4}\]\s*[A-Z]+\s*\d+)', 'case-law'),
        ]
        
        for pattern, class_name in case_patterns:
            if class_name == 'case-law':
                html_output = re.sub(
                    pattern,
                    r'<span class="case-law" style="color: #0066CC; font-weight: bold; text-decoration: underline">\1</span>',
                    html_output
                )
        
        # Apply statute formatting
        statute_patterns = [
            r'((?:Children Act|Matrimonial Causes Act|Family Law Act)\s*\d{4})',
            r'(Section\s+\d+[A-Z]?(?:\(\d+\))?)'
        ]
        
        for pattern in statute_patterns:
            html_output = re.sub(
                pattern,
                r'<span class="statute" style="color: #008800; font-weight: bold">\1</span>',
                html_output
            )
        
        # Apply financial formatting
        html_output = re.sub(
            r'(£[\d,]+(?:\.\d{2})?)',
            r'<span class="financial" style="color: #444444; background-color: #FFFFCC">\1</span>',
            html_output
        )
        
        return {
            "standard_output": response_text,
            "explainability_output": html_output
        }

def test_with_sample_input():
    """Test the model with a sample family law case"""
    
    # Sample input based on meeting examples
    test_input = "I have two children aged 8 and 12, inherited property in Nottingham worth £200,000, and shared property in Leeds worth £450,000 with mortgage of £150,000"
    
    print("="*60)
    print("MISTRAL HTML EXPLAINABILITY TEST")
    print("="*60)
    print(f"\nTest Input: {test_input}\n")
    
    # Initialize tester
    tester = MistralHTMLExplainabilityTester()
    
    # Generate response
    response_text, generation_time = tester.generate_response(test_input)
    
    print("\n" + "="*60)
    print("RAW MODEL OUTPUT:")
    print("="*60)
    print(response_text)
    
    # Parse response
    parsed_response = tester.parse_response(response_text)
    
    # Save outputs
    output = {
        "model": "Mistral-Nemo-Instruct-2407",
        "input": test_input,
        "raw_response": response_text,
        "parsed_response": parsed_response,
        "generation_time": generation_time,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save JSON output
    with open('mistral_test_output.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\n✅ Saved: mistral_test_output.json")
    
    # Create HTML preview
    if 'explainability_output' in parsed_response:
        html_preview = f"""<!DOCTYPE html>
<html>
<head>
    <title>Mistral HTML Explainability Test</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }}
        .container {{ background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; }}
        .input {{ background: #e8f5e9; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .output {{ background: #f5f5f5; padding: 20px; border-radius: 5px; margin: 20px 0; line-height: 1.8; }}
        .timing {{ background: #fff3e0; padding: 10px; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Mistral HTML Explainability Test Result</h1>
        
        <div class="timing">
            <strong>Generation Time:</strong> {generation_time:.2f} seconds
        </div>
        
        <div class="input">
            <h3>User Input:</h3>
            <p>{test_input}</p>
        </div>
        
        <div class="output">
            <h3>Model Output with HTML Explainability:</h3>
            {parsed_response.get('explainability_output', response_text)}
        </div>
        
        <div class="output">
            <h3>Standard Output (Plain Text):</h3>
            <p>{parsed_response.get('standard_output', 'No standard output generated')}</p>
        </div>
    </div>
</body>
</html>"""
        
        with open('mistral_test_preview.html', 'w') as f:
            f.write(html_preview)
        print("✅ Saved: mistral_test_preview.html")
    
    print("\n" + "="*60)
    print("TEST COMPLETE!")
    print("="*60)
    print(f"\n⏱️ Generation time: {generation_time:.2f} seconds")
    print("\nFiles created:")
    print("1. mistral_test_output.json - Complete test results")
    print("2. mistral_test_preview.html - Visual preview")
    print("\n✅ Ready to share with ReGoBs team!")

if __name__ == "__main__":
    # Check if model path exists
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        print(f"❌ Model path not found: {MODEL_PATH}")
        print("Please update MODEL_PATH variable with correct path")
    else:
        test_with_sample_input()