#!/usr/bin/env python3
"""
Complete XML Structure Analyzer - GUARANTEED to process ALL files
"""

import os
import glob
import re
from collections import Counter, defaultdict
import json
from datetime import datetime
import traceback

def analyze_file_structure(filepath):
    """Extract key structural elements from a single file"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        result = {
            'filename': os.path.basename(filepath),
            'filepath': filepath,
            'size_kb': len(content) // 1024,
            'has_judgmentBody': '<judgmentBody>' in content,
            'has_decision': '<decision>' in content,
            'has_order': '<order>' in content or 'consent order' in content.lower(),
            'paragraph_count': content.count('<paragraph'),
            'num_tags': content.count('<num>'),
            'level_tags': content.count('<level'),
            'quote_blocks': content.count('<block name="embeddedStructure"'),
            'citations': content.count('uk:cite>'),
            'processed': True,
            'error': None
        }
        
        # Detect document type
        if 'consent order' in content.lower():
            result['doc_type'] = 'consent_order'
        elif '<judgmentBody>' in content:
            result['doc_type'] = 'judgment'
        elif '<order>' in content:
            result['doc_type'] = 'order'
        elif '<decision>' in content:
            result['doc_type'] = 'decision'
        else:
            result['doc_type'] = 'other'
        
        # Extract court from metadata
        court_match = re.search(r'uk:court>([^<]+)<', content)
        result['court'] = court_match.group(1) if court_match else 'unknown'
        
        # Extract year
        year_match = re.search(r'uk:year>([^<]+)<', content)
        result['year'] = year_match.group(1) if year_match else 'unknown'
        
        # Check how paragraphs start
        para_starts = re.findall(r'<paragraph[^>]*>.*?<num>([^<]+)</num>', content[:50000])
        result['para_numbering'] = para_starts[:3] if para_starts else []
        
        # Check main structural tags
        result['has_header'] = '<header>' in content
        result['has_coverPage'] = '<coverPage>' in content
        
        return result
        
    except Exception as e:
        # If ANY error occurs, we still record the file
        return {
            'filename': os.path.basename(filepath),
            'filepath': filepath,
            'processed': False,
            'error': str(e),
            'doc_type': 'error',
            'paragraph_count': 0
        }

def main():
    xml_dir = "/users/bgxp240/ailes_legal_ai/data/raw/xml_judgments"
    
    # Get ALL XML files
    files = glob.glob(os.path.join(xml_dir, "*.xml"))
    
    print(f"{'='*60}")
    print(f"COMPLETE STRUCTURAL ANALYSIS")
    print(f"{'='*60}")
    print(f"Found {len(files)} XML files to process")
    print(f"Started: {datetime.now()}\n")
    
    # VERIFICATION: List files to make sure we got them all
    print(f"First 5 files found:")
    for f in files[:5]:
        print(f"  - {os.path.basename(f)}")
    print(f"Last 5 files found:")
    for f in files[-5:]:
        print(f"  - {os.path.basename(f)}")
    
    # Process ALL files with tracking
    all_results = []
    processed_count = 0
    error_count = 0
    doc_types = Counter()
    courts = Counter()
    years = Counter()
    para_counts = []
    numbering_patterns = Counter()
    
    print(f"\nProcessing ALL {len(files)} files...")
    print("Progress:")
    
    for i, filepath in enumerate(files, 1):
        # Progress indicator every 100 files
        if i % 100 == 0:
            print(f"  [{i}/{len(files)}] {i/len(files)*100:.1f}% complete...")
        
        # Analyze this file
        result = analyze_file_structure(filepath)
        all_results.append(result)
        
        if result.get('processed', False):
            processed_count += 1
            doc_types[result['doc_type']] += 1
            courts[result.get('court', 'unknown')] += 1
            years[result.get('year', 'unknown')] += 1
            para_counts.append(result.get('paragraph_count', 0))
            
            if result.get('para_numbering'):
                pattern = '-'.join(result['para_numbering'][:2])
                numbering_patterns[pattern] += 1
        else:
            error_count += 1
            print(f"  ERROR in file {i}: {result['filename']} - {result.get('error', 'Unknown error')}")
    
    # VERIFICATION CHECK
    print(f"\n{'='*60}")
    print(f"PROCESSING VERIFICATION:")
    print(f"  Files found: {len(files)}")
    print(f"  Files analyzed: {len(all_results)}")
    print(f"  Successfully processed: {processed_count}")
    print(f"  Errors: {error_count}")
    
    if len(files) != len(all_results):
        print(f"  WARNING: Mismatch! {len(files) - len(all_results)} files not processed!")
    else:
        print(f"  ✓ ALL FILES PROCESSED!")
    
    # Generate report
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE - SUMMARY REPORT")
    print(f"{'='*60}")
    
    print(f"\n1. DOCUMENT TYPES (from {processed_count} successful files):")
    print("-" * 30)
    for doc_type, count in doc_types.most_common():
        percentage = (count/processed_count)*100 if processed_count > 0 else 0
        print(f"  {doc_type:15} : {count:5} files ({percentage:5.1f}%)")
    
    print(f"\n2. DOCUMENT STRUCTURE:")
    print("-" * 30)
    has_judgment_body = sum(1 for r in all_results if r.get('has_judgmentBody', False))
    has_decision = sum(1 for r in all_results if r.get('has_decision', False))
    has_order = sum(1 for r in all_results if r.get('has_order', False))
    has_header = sum(1 for r in all_results if r.get('has_header', False))
    has_cover = sum(1 for r in all_results if r.get('has_coverPage', False))
    
    print(f"  Files with <judgmentBody>: {has_judgment_body} ({has_judgment_body/len(files)*100:.1f}%)")
    print(f"  Files with <decision>: {has_decision} ({has_decision/len(files)*100:.1f}%)")
    print(f"  Files with <order>: {has_order} ({has_order/len(files)*100:.1f}%)")
    print(f"  Files with <header>: {has_header} ({has_header/len(files)*100:.1f}%)")
    print(f"  Files with <coverPage>: {has_cover} ({has_cover/len(files)*100:.1f}%)")
    
    if para_counts:
        print(f"\n3. PARAGRAPH STATISTICS:")
        print("-" * 30)
        avg_paras = sum(para_counts) / len(para_counts)
        print(f"  Average paragraphs per file: {avg_paras:.1f}")
        print(f"  Min paragraphs: {min(para_counts)}")
        print(f"  Max paragraphs: {max(para_counts)}")
        
        # Paragraph distribution
        print(f"\n  Paragraph count distribution:")
        ranges = [(0, 10), (11, 50), (51, 100), (101, 200), (201, 500), (501, 10000)]
        for low, high in ranges:
            count = sum(1 for p in para_counts if low <= p <= high)
            print(f"    {low:3}-{high:4} paras: {count:4} files ({count/len(para_counts)*100:5.1f}%)")
    
    print(f"\n4. PARAGRAPH NUMBERING PATTERNS (top 10):")
    print("-" * 30)
    for pattern, count in numbering_patterns.most_common(10):
        print(f"  Pattern '{pattern}': {count} files")
    
    print(f"\n5. COURTS DISTRIBUTION (top 10):")
    print("-" * 30)
    for court, count in courts.most_common(10):
        if court != 'unknown':
            print(f"  {court:20} : {count:5} files")
    
    print(f"\n6. YEAR DISTRIBUTION:")
    print("-" * 30)
    year_sorted = sorted([(y, c) for y, c in years.items() if y != 'unknown'])
    for year, count in year_sorted:
        print(f"  {year}: {count:4} files")
    
    # Save detailed results
    print(f"\n7. SAVING DETAILED RESULTS...")
    print("-" * 30)
    
    # CSV with ALL files
    csv_filename = f"xml_analysis_ALL_{len(files)}_files_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_filename, 'w') as f:
        f.write("filename,doc_type,court,year,paragraphs,has_judgmentBody,has_decision,has_order,size_kb,processed,error\n")
        for r in all_results:
            f.write(f"{r['filename']},{r.get('doc_type','unknown')},{r.get('court','unknown')},"
                   f"{r.get('year','unknown')},{r.get('paragraph_count',0)},"
                   f"{r.get('has_judgmentBody',False)},{r.get('has_decision',False)},"
                   f"{r.get('has_order',False)},{r.get('size_kb',0)},"
                   f"{r.get('processed',False)},{r.get('error','')}\n")
    
    print(f"  ✓ Detailed CSV saved: {csv_filename}")
    
    # Save error files list if any
    if error_count > 0:
        error_file = f"error_files_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(error_file, 'w') as f:
            for r in all_results:
                if not r.get('processed', False):
                    f.write(f"{r['filename']}: {r.get('error', 'Unknown')}\n")
        print(f"  ✓ Error files list saved: {error_file}")
    
    # Show examples
    print(f"\n8. SAMPLE FILES BY TYPE:")
    print("-" * 30)
    for doc_type in doc_types.keys():
        samples = [r['filename'] for r in all_results if r.get('doc_type') == doc_type][:2]
        if samples:
            print(f"  {doc_type}:")
            for s in samples:
                print(f"    - {s}")
    
    print(f"\n{'='*60}")
    print(f"FINAL VERIFICATION:")
    print(f"  Total files in directory: {len(files)}")
    print(f"  Total files processed: {len(all_results)}")
    print(f"  Match: {'YES ✓' if len(files) == len(all_results) else 'NO ✗'}")
    print(f"{'='*60}")
    
    print(f"\nCompleted: {datetime.now()}")
    
    return all_results

if __name__ == "__main__":
    results = main()