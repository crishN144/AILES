#!/usr/bin/env python3
"""
AILES Training Data Quality Validator
Beautiful analysis and validation of your training data
"""

import json
from datetime import datetime
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict
import statistics
import re

class TrainingDataValidator:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.components = ['chatbot', 'predictor', 'explainer']
        
    def load_data(self, component: str) -> List[Dict[str, Any]]:
        """Load training data for a component"""
        data_file = self.data_dir / f"{component}_training_data.jsonl"
        
        if not data_file.exists():
            print(f"❌ {component}_training_data.jsonl not found")
            return []
        
        examples = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    example = json.loads(line.strip())
                    examples.append(example)
                except json.JSONDecodeError as e:
                    print(f"⚠️  JSON error in {component} line {line_num}: {e}")
                    continue
        
        return examples
    
    def analyze_component_overview(self, component: str) -> Dict[str, Any]:
        """Get overview statistics for a component"""
        examples = self.load_data(component)
        
        if not examples:
            return {'error': 'No data loaded'}
        
        # Basic stats
        stats = {
            'total_examples': len(examples),
            'instruction_lengths': [],
            'input_lengths': [],
            'output_lengths': [],
            'case_types': Counter(),
            'complexity_scores': [],
            'confidence_scores': []
        }
        
        for example in examples:
            stats['instruction_lengths'].append(len(example.get('instruction', '')))
            stats['input_lengths'].append(len(example.get('input', '')))
            stats['output_lengths'].append(len(example.get('output', '')))
            
            metadata = example.get('metadata', {})
            if 'case_type' in metadata:
                stats['case_types'][metadata['case_type']] += 1
            if 'complexity_score' in metadata:
                stats['complexity_scores'].append(metadata['complexity_score'])
            if 'confidence' in metadata:
                stats['confidence_scores'].append(metadata['confidence'])
        
        # Calculate summary stats
        summary = {
            'total_examples': stats['total_examples'],
            'avg_instruction_length': statistics.mean(stats['instruction_lengths']) if stats['instruction_lengths'] else 0,
            'avg_input_length': statistics.mean(stats['input_lengths']) if stats['input_lengths'] else 0,
            'avg_output_length': statistics.mean(stats['output_lengths']) if stats['output_lengths'] else 0,
            'case_type_distribution': dict(stats['case_types'].most_common()),
            'avg_complexity': statistics.mean(stats['complexity_scores']) if stats['complexity_scores'] else 0,
            'avg_confidence': statistics.mean(stats['confidence_scores']) if stats['confidence_scores'] else 0
        }
        
        return summary
    
    def show_beautiful_overview(self):
        """Display a beautiful overview of all training data"""
        print("🎯" + "="*80)
        print("📊 AILES TRAINING DATA QUALITY ANALYSIS")
        print("="*82)
        
        total_examples = 0
        
        for component in self.components:
            print(f"\n🤖 {component.upper()} COMPONENT")
            print("-" * 50)
            
            summary = self.analyze_component_overview(component)
            
            if 'error' in summary:
                print(f"❌ {summary['error']}")
                continue
            
            total_examples += summary['total_examples']
            
            # Basic metrics
            print(f"📈 Total Examples: {summary['total_examples']:,}")
            print(f"📝 Avg Instruction Length: {summary['avg_instruction_length']:.0f} chars")
            print(f"📥 Avg Input Length: {summary['avg_input_length']:.0f} chars")  
            print(f"📤 Avg Output Length: {summary['avg_output_length']:.0f} chars")
            print(f"🎯 Avg Complexity Score: {summary['avg_complexity']:.2f}")
            print(f"🔍 Avg Confidence Score: {summary['avg_confidence']:.2f}")
            
            # Case type distribution
            print(f"\n📋 Case Type Distribution:")
            for case_type, count in summary['case_type_distribution'].items():
                percentage = (count / summary['total_examples']) * 100
                bar = "█" * min(20, int(percentage / 5))
                print(f"   {case_type:20} │{bar:<20}│ {count:4} ({percentage:5.1f}%)")
        
        print(f"\n🎊 TOTAL TRAINING EXAMPLES: {total_examples:,}")
        print("="*82)
    
    def show_sample_examples(self, component: str, num_samples: int = 3):
        """Show sample examples with beautiful formatting"""
        examples = self.load_data(component)
        
        if not examples:
            print(f"❌ No {component} examples to show")
            return
        
        print(f"\n🔍 SAMPLE {component.upper()} EXAMPLES")
        print("="*80)
        
        # Show examples from different case types
        case_types = set()
        samples = []
        
        for example in examples:
            case_type = example.get('metadata', {}).get('case_type', 'unknown')
            if case_type not in case_types and len(samples) < num_samples:
                case_types.add(case_type)
                samples.append(example)
        
        # If we don't have enough variety, just take first N
        if len(samples) < num_samples:
            samples = examples[:num_samples]
        
        for i, example in enumerate(samples, 1):
            print(f"\n📝 EXAMPLE {i}")
            print("-" * 40)
            
            # Metadata
            metadata = example.get('metadata', {})
            print(f"🏷️  Case Type: {metadata.get('case_type', 'unknown')}")
            print(f"📊 Complexity: {metadata.get('complexity_score', 0):.2f}")
            print(f"🎯 Confidence: {metadata.get('confidence', 0):.2f}")
            
            # Instruction (truncated)
            instruction = example.get('instruction', '')
            print(f"\n💡 INSTRUCTION:")
            print(f"   {self._truncate_text(instruction, 200)}")
            
            # Input (truncated)  
            input_text = example.get('input', '')
            print(f"\n📥 INPUT:")
            print(f"   {self._truncate_text(input_text, 300)}")
            
            # Output (truncated and formatted)
            output_text = example.get('output', '')
            print(f"\n📤 OUTPUT:")
            try:
                # Try to parse as JSON for pretty printing
                output_json = json.loads(output_text)
                formatted_output = json.dumps(output_json, indent=2)[:500]
                if len(formatted_output) >= 500:
                    formatted_output += "..."
                print(f"   {formatted_output}")
            except json.JSONDecodeError:
                print(f"   {self._truncate_text(output_text, 400)}")
    
    def validate_json_quality(self, component: str) -> Dict[str, Any]:
        """Validate JSON structure and quality"""
        examples = self.load_data(component)
        
        validation_results = {
            'total_examples': len(examples),
            'valid_json_outputs': 0,
            'structured_outputs': 0,
            'common_output_fields': Counter(),
            'quality_issues': []
        }
        
        for i, example in enumerate(examples):
            # Check output JSON validity
            output = example.get('output', '')
            try:
                output_json = json.loads(output)
                validation_results['valid_json_outputs'] += 1
                
                # Check if it's a structured response (dict)
                if isinstance(output_json, dict):
                    validation_results['structured_outputs'] += 1
                    
                    # Count common fields
                    for field in output_json.keys():
                        validation_results['common_output_fields'][field] += 1
                
            except json.JSONDecodeError:
                validation_results['quality_issues'].append({
                    'example_index': i,
                    'issue': 'Invalid JSON in output',
                    'preview': output[:100] + "..." if len(output) > 100 else output
                })
        
        return validation_results
    
    def show_quality_report(self, component: str):
        """Show detailed quality report for a component"""
        print(f"\n🔍 DETAILED QUALITY REPORT: {component.upper()}")
        print("="*80)
        
        validation = self.validate_json_quality(component)
        
        total = validation['total_examples']
        valid_json = validation['valid_json_outputs']
        structured = validation['structured_outputs']
        
        print(f"📊 JSON Validity: {valid_json}/{total} ({(valid_json/total*100):.1f}%)")
        print(f"🏗️  Structured Outputs: {structured}/{total} ({(structured/total*100):.1f}%)")
        
        if validation['common_output_fields']:
            print(f"\n📋 Common Output Fields:")
            for field, count in validation['common_output_fields'].most_common(10):
                percentage = (count / structured) * 100 if structured > 0 else 0
                print(f"   {field:20} │ {count:4}/{structured:4} ({percentage:5.1f}%)")
        
        if validation['quality_issues']:
            print(f"\n⚠️  Quality Issues ({len(validation['quality_issues'])}):")
            for issue in validation['quality_issues'][:5]:  # Show first 5
                print(f"   Example {issue['example_index']}: {issue['issue']}")
                print(f"      Preview: {issue['preview']}")
    
    def search_examples(self, component: str, search_term: str, max_results: int = 5):
        """Search for examples containing specific terms"""
        examples = self.load_data(component)
        
        matches = []
        search_lower = search_term.lower()
        
        for i, example in enumerate(examples):
            # Search in all text fields
            searchable_text = ' '.join([
                example.get('instruction', ''),
                example.get('input', ''),
                example.get('output', ''),
                str(example.get('metadata', {}))
            ]).lower()
            
            if search_lower in searchable_text:
                matches.append((i, example))
                if len(matches) >= max_results:
                    break
        
        print(f"\n🔍 SEARCH RESULTS for '{search_term}' in {component.upper()}")
        print(f"Found {len(matches)} matches (showing first {min(len(matches), max_results)})")
        print("="*80)
        
        for idx, (example_idx, example) in enumerate(matches):
            print(f"\n📝 MATCH {idx+1} (Example #{example_idx})")
            print("-" * 40)
            
            case_type = example.get('metadata', {}).get('case_type', 'unknown')
            print(f"🏷️  Case Type: {case_type}")
            
            # Show relevant excerpts
            input_text = example.get('input', '')
            if search_lower in input_text.lower():
                print(f"📥 INPUT MATCH:")
                print(f"   ...{self._get_context_around_term(input_text, search_term, 150)}...")
            
            output_text = example.get('output', '')
            if search_lower in output_text.lower():
                print(f"📤 OUTPUT MATCH:")  
                print(f"   ...{self._get_context_around_term(output_text, search_term, 200)}...")
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text with ellipsis"""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."
    
    def _get_context_around_term(self, text: str, term: str, context_length: int) -> str:
        """Get context around a search term"""
        text_lower = text.lower()
        term_lower = term.lower()
        
        pos = text_lower.find(term_lower)
        if pos == -1:
            return text[:context_length]
        
        start = max(0, pos - context_length // 2)
        end = min(len(text), pos + len(term) + context_length // 2)
        
        context = text[start:end]
        if start > 0:
            context = "..." + context
        if end < len(text):
            context = context + "..."
            
        return context
    
    def generate_report_summary(self) -> str:
        """Generate a summary report"""
        report = []
        report.append("📊 AILES TRAINING DATA SUMMARY REPORT")
        report.append("="*50)
        report.append(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        total_examples = 0
        for component in self.components:
            summary = self.analyze_component_overview(component)
            if 'error' not in summary:
                total_examples += summary['total_examples']
                report.append(f"\n🤖 {component.upper()}: {summary['total_examples']:,} examples")
                
                # Top case types
                top_cases = list(summary['case_type_distribution'].items())[:3]
                for case_type, count in top_cases:
                    percentage = (count / summary['total_examples']) * 100
                    report.append(f"   • {case_type}: {count} ({percentage:.1f}%)")
        
        report.append(f"\n🎊 TOTAL: {total_examples:,} training examples")
        report.append(f"✅ Ready for LLaMA fine-tuning!")
        
        return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description="Validate AILES training data quality")
    parser.add_argument("--data_dir", default="data/processed_full_ultra", 
                       help="Directory containing training data")
    parser.add_argument("--component", choices=['chatbot', 'predictor', 'explainer'], 
                       help="Specific component to analyze")
    parser.add_argument("--samples", type=int, default=3, 
                       help="Number of sample examples to show")
    parser.add_argument("--search", help="Search for examples containing specific term")
    parser.add_argument("--quality", action="store_true", 
                       help="Show detailed quality report")
    parser.add_argument("--overview", action="store_true", 
                       help="Show overview of all components")
    
    args = parser.parse_args()
    
    validator = TrainingDataValidator(args.data_dir)
    
    if args.overview or not any([args.component, args.search, args.quality]):
        validator.show_beautiful_overview()
    
    if args.component:
        if args.quality:
            validator.show_quality_report(args.component)
        elif args.search:
            validator.search_examples(args.component, args.search)
        else:
            validator.show_sample_examples(args.component, args.samples)
    
    if args.search and not args.component:
        # Search all components
        for component in validator.components:
            validator.search_examples(component, args.search, 2)
    
    # Always show summary at the end
    print("\n" + validator.generate_report_summary())

if __name__ == "__main__":
    main()