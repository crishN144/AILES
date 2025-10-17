#!/usr/bin/env python3
import os
import re
from pathlib import Path

def analyze_unknown_cases():
    """Deep dive into the 'unknown' cases to find hidden financial remedy cases"""
    
    # Read the original analysis to get unknown case files
    xml_dir = Path('/users/bgxp240/ailes_legal_ai/data/raw/xml_judgments/')
    xml_files = list(xml_dir.glob('*.xml'))
    
    # Enhanced financial keywords (broader search)
    primary_financial = [
        'form e', 'section 25', 'matrimonial causes act', 
        'financial remedy', 'ancillary relief'
    ]
    
    secondary_financial = [
        'lump sum', 'property adjustment', 'pension sharing',
        'periodical payments', 'matrimonial property', 'spousal maintenance',
        'clean break', 'freezing order', 'without notice',
        'divorce settlement', 'matrimonial assets', 'financial order'
    ]
    
    tertiary_financial = [
        'white v white', 'miller v miller', 'mcfarlane',
        'sharing principle', 'needs compensation', 'non-matrimonial',
        'husband', 'wife', 'petitioner', 'respondent', 'marriage',
        'divorce', 'separation', 'matrimonial home'
    ]
    
    # Care proceedings indicators (to exclude)
    care_indicators = [
        'care order', 'care proceedings', 'local authority',
        'child protection', 'threshold criteria', 'interim care',
        'special guardianship', 'adoption order', 'placement order'
    ]
    
    print("Analyzing unknown cases for hidden financial content...")
    
    recovered_financial = []
    borderline_cases = []
    
    for i, xml_file in enumerate(xml_files):
        if i % 200 == 0:
            print(f"Processed {i}/{len(xml_files)} files...")
        
        try:
            with open(xml_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().lower()
            
            # Count different types of keywords
            primary_count = sum(1 for kw in primary_financial if kw in content)
            secondary_count = sum(1 for kw in secondary_financial if kw in content)
            tertiary_count = sum(1 for kw in tertiary_financial if kw in content)
            care_count = sum(1 for kw in care_indicators if kw in content)
            
            # Skip if clearly care proceedings
            if care_count >= 3:
                continue
                
            # Skip if adoption-focused
            if 'adoption' in content and 'financial' not in content:
                continue
                
            total_financial_score = primary_count * 3 + secondary_count * 2 + tertiary_count
            
            # Classification logic
            filename = os.path.basename(xml_file)
            word_count = len(content.split())
            
            case_info = {
                'filename': filename,
                'primary_keywords': primary_count,
                'secondary_keywords': secondary_count, 
                'tertiary_keywords': tertiary_count,
                'care_keywords': care_count,
                'total_score': total_financial_score,
                'word_count': word_count
            }
            
            # Recovered financial remedy cases (likely missed)
            if (primary_count >= 1 and secondary_count >= 2) or total_financial_score >= 8:
                if care_count < 2:  # Not primarily care proceedings
                    recovered_financial.append(case_info)
            
            # Borderline cases worth reviewing
            elif (primary_count >= 1 or secondary_count >= 3) and care_count < 2:
                borderline_cases.append(case_info)
                
        except Exception as e:
            continue
    
    print(f"\n{'='*60}")
    print("DEEP ANALYSIS RESULTS")
    print(f"{'='*60}")
    
    print(f"\nRECOVERED FINANCIAL REMEDY CASES: {len(recovered_financial)}")
    if recovered_financial:
        print("Top 10 recovered cases:")
        recovered_financial.sort(key=lambda x: x['total_score'], reverse=True)
        for case in recovered_financial[:10]:
            print(f"  {case['filename'][:45]}... "
                  f"Score: {case['total_score']}, "
                  f"Primary: {case['primary_keywords']}, "
                  f"Secondary: {case['secondary_keywords']}")
    
    print(f"\nBORDERLINE CASES: {len(borderline_cases)}")
    if borderline_cases:
        print("Top 5 borderline cases:")
        borderline_cases.sort(key=lambda x: x['total_score'], reverse=True)
        for case in borderline_cases[:5]:
            print(f"  {case['filename'][:45]}... "
                  f"Score: {case['total_score']}")
    
    # Save detailed results
    with open('/users/bgxp240/ailes_legal_ai/recovered_financial_cases.txt', 'w') as f:
        f.write("RECOVERED FINANCIAL REMEDY CASES\n")
        f.write("="*50 + "\n\n")
        
        for case in recovered_financial:
            f.write(f"File: {case['filename']}\n")
            f.write(f"Total Score: {case['total_score']}\n")
            f.write(f"Primary Keywords: {case['primary_keywords']}\n")
            f.write(f"Secondary Keywords: {case['secondary_keywords']}\n")
            f.write(f"Tertiary Keywords: {case['tertiary_keywords']}\n")
            f.write(f"Word Count: {case['word_count']}\n")
            f.write("-" * 30 + "\n")
    
    print(f"\nSUMMARY:")
    original_count = 611
    recovered_count = len(recovered_financial)
    total_financial = original_count + recovered_count
    
    print(f"Original financial remedy cases: {original_count}")
    print(f"Recovered financial remedy cases: {recovered_count}")
    print(f"TOTAL FINANCIAL REMEDY CASES: {total_financial}")
    
    if total_financial >= 800:
        print("🎉 EXCELLENT: Strong dataset for AI training!")
    elif total_financial >= 700:
        print("✅ GOOD: Sufficient for initial training")
    else:
        print("⚠️  MARGINAL: Consider BAILII scraping for more cases")
    
    return recovered_financial, borderline_cases

if __name__ == "__main__":
    recovered, borderline = analyze_unknown_cases()
