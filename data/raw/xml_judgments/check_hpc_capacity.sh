#!/bin/bash

echo "==================================================="
echo "HPC AIRE Storage and Compute Capacity Check"
echo "==================================================="
echo ""

echo "1. HOME DIRECTORY USAGE:"
echo "------------------------"
quota -u $USER 2>/dev/null || echo "Quota command not available"
df -h $HOME
echo ""

echo "2. PROJECT DIRECTORY USAGE:"
echo "---------------------------"
du -sh /users/bgxp240/ailes_legal_ai/ 2>/dev/null || echo "Project directory not found"
echo ""

echo "3. SCRATCH/TEMPORARY SPACE:"
echo "---------------------------"
df -h /nobackup 2>/dev/null || echo "No /nobackup available"
df -h /tmp
echo ""

echo "4. COMPUTE NODE SPECIFICATIONS:"
echo "-------------------------------"
sinfo -o "%P %c %m %t %N" | head -5
echo ""

echo "5. AVAILABLE PARTITIONS:"
echo "------------------------"
scontrol show partition | grep -E "PartitionName|MaxTime|MaxNodes|DefMemPerCPU" | head -20
echo ""

echo "6. CURRENT RUNNING JOBS:"
echo "------------------------"
squeue -u $USER
echo ""

echo "7. MODEL SIZE RECOMMENDATIONS:"
echo "------------------------------"
echo "Available home space: $(df -h $HOME | tail -1 | awk '{print $4}')"
echo ""
echo "Recommended model sizes based on available space:"
home_avail=$(df --output=avail $HOME | tail -1)
if [ $home_avail -gt 50000000 ]; then
    echo "✅ Can accommodate: Up to Llama 70B (50+ GB available)"
elif [ $home_avail -gt 25000000 ]; then
    echo "✅ Can accommodate: Up to Llama 11B (25+ GB available)" 
elif [ $home_avail -gt 15000000 ]; then
    echo "✅ Can accommodate: Up to Llama 7B (15+ GB available)"
elif [ $home_avail -gt 8000000 ]; then
    echo "✅ Can accommodate: Llama 3B only (8+ GB available)"
else
    echo "❌ Insufficient space for larger models"
fi
echo ""

echo "8. MEMORY RECOMMENDATIONS:"
echo "-------------------------"
max_mem=$(sinfo -o "%m" | tail -n +2 | sort -nr | head -1)
echo "Maximum node memory: ${max_mem} MB"
if [ $max_mem -gt 128000 ]; then
    echo "✅ Sufficient memory for Llama 70B"
elif [ $max_mem -gt 64000 ]; then
    echo "✅ Sufficient memory for Llama 11B" 
elif [ $max_mem -gt 32000 ]; then
    echo "✅ Sufficient memory for Llama 7B"
else
    echo "⚠️  Limited to smaller models"
fi

