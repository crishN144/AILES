#!/usr/bin/env python3
import os
import re
from pathlib import Path

def analyze_xml_file(filepath):
    """Analyze a single XML file for financial remedy content"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read().lower()
        
        # Financial remedy keywords (must have multiple to count)
        financial_keywords = [
            'form e', 'section 25', 'matrimonial causes act', 
            'financial remedy', 'ancillary relief', 'lump sum',
            'property adjustment', 'pension sharing', 'periodical payments',
            'white v white', 'miller v miller', 'clean break',
            'matrimonial property', 'non-matrimonial', 'sharing principle',
            'needs compensation', 'freezing order', 'without notice'
        ]
        
        # Care/child protection keywords (indicates NOT financial remedy)
        care_keywords = [
            'care order', 'care proceedings', 'local authority',
            'child protection', 'threshold criteria', 'care plan',
            'adoption order', 'placement order', 'special guardianship',
            'interim care order', 'supervision order'
        ]
        
        # Count keyword matches
        financial_count = sum(1 for keyword in financial_keywords if keyword in content)
        care_count = sum(1 for keyword in care_keywords if keyword in content)
        
        # Determine case type
        case_type = "unknown"
        if financial_count >= 3 and care_count < 2:
            case_type = "financial_remedy"
        elif care_count >= 2:
            case_type = "care_proceedings"
        elif 'adoption' in content and 'financial remedy' not in content:
            case_type = "adoption"
        elif 'court of protection' in content or 'mental capacity' in content:
            case_type = "mental_capacity"
        
        return {
            'filename': os.path.basename(filepath),
            'case_type': case_type,
            'financial_keywords': financial_count,
            'care_keywords': care_count,
            'word_count': len(content.split())
        }
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def main():
    xml_dir = Path('/users/bgxp240/ailes_legal_ai/data/raw/xml_judgments/')
    
    # Get all XML files
    xml_files = list(xml_dir.glob('*.xml'))
    total_files = len(xml_files)
    
    print(f"Found {total_files} XML files")
    print("Analyzing case types...\n")
    
    # Analyze each file
    results = []
    financial_remedy_cases = []
    
    for i, xml_file in enumerate(xml_files):
        if i % 100 == 0:  # Progress indicator
            print(f"Processed {i}/{total_files} files...")
            
        result = analyze_xml_file(xml_file)
        if result:
            results.append(result)
            if result['case_type'] == 'financial_remedy':
                financial_remedy_cases.append(result)
    
    # Summary statistics
    case_types = {}
    for result in results:
        case_type = result['case_type']
        case_types[case_type] = case_types.get(case_type, 0) + 1
    
    print(f"\n{'='*50}")
    print("REALITY CHECK RESULTS")
    print(f"{'='*50}")
    print(f"Total XML files analyzed: {len(results)}")
    print(f"\nCase Type Breakdown:")
    for case_type, count in sorted(case_types.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(results)) * 100
        print(f"  {case_type.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
    
    print(f"\n{'='*30}")
    print("FINANCIAL REMEDY CASES")
    print(f"{'='*30}")
    print(f"True financial remedy cases: {len(financial_remedy_cases)}")
    
    if financial_remedy_cases:
        print(f"\nTop 10 Financial Remedy Cases:")
        # Sort by keyword count (quality indicator)
        top_cases = sorted(financial_remedy_cases, 
                          key=lambda x: x['financial_keywords'], reverse=True)[:10]
        
        for case in top_cases:
            print(f"  {case['filename'][:50]}... "
                  f"(Keywords: {case['financial_keywords']}, "
                  f"Words: {case['word_count']})")
    
    # Save detailed results
    output_file = '/users/bgxp240/ailes_legal_ai/financial_remedy_analysis.txt'
    with open(output_file, 'w') as f:
        f.write("DETAILED FINANCIAL REMEDY CASE ANALYSIS\n")
        f.write("="*50 + "\n\n")
        
        for case in financial_remedy_cases:
            f.write(f"File: {case['filename']}\n")
            f.write(f"Financial Keywords: {case['financial_keywords']}\n")
            f.write(f"Word Count: {case['word_count']}\n")
            f.write("-" * 30 + "\n")
    
    print(f"\nDetailed analysis saved to: {output_file}")
    print(f"\nRECOMMENDATION:")
    if len(financial_remedy_cases) < 500:
        print("❌ INSUFFICIENT financial remedy cases for AI training")
        print("✅ RECOMMEND: Scrape BAILII for additional financial cases")
    else:
        print("✅ SUFFICIENT financial remedy cases for initial training")

if __name__ == "__main__":
    main()
