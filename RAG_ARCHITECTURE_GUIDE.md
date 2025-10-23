# RAG Architecture Guide for AILES + Fine-tuned Llama-3.1-8B

## 🎯 Overview: Hybrid Approach

Your **fine-tuned Llama-3.1-8B** is excellent at:
- ✅ Extracting facts from provided judgment text
- ✅ Understanding UK family law terminology
- ✅ Citing paragraph numbers
- ✅ Concise, factual responses

**RAG will ADD:**
- ✅ Access to full judgment corpus (not just excerpts)
- ✅ Multi-document reasoning
- ✅ Finding relevant cases automatically
- ✅ Up-to-date information (new judgments)
- ✅ Comprehensive Form E guidance
- ✅ Statutory references

---

## 🏗️ Recommended RAG Architecture

```
USER QUERY: "What are typical lump sum orders in high-value cases?"
                              ↓
                    [QUERY ROUTER]
                              ↓
                    ┌─────────┴─────────┐
                    ↓                   ↓
            [RAG RETRIEVAL]     [FINE-TUNED LLAMA]
         (Vector DB Search)      (Direct Query)
                    ↓                   ↓
            [FINE-TUNED LLAMA-3.1-8B]
         (Synthesis + Extraction)
                    ↓
            [FINAL RESPONSE with citations]
```

---

## 📚 What to Put in Vector Database

### ✅ YES - Use Your XML Judgment Files!

**Source 1: UK Family Law Judgments**
- Location: `/users/bgxp240/ailes_legal_ai/data/raw/xml_judgments/`
- Count: 4,611 XML files
- Size: 413 MB
- Content: Full text of judgments

**Source 2: UK Family Law Statutes** (You already have these)
- Matrimonial Causes Act 1973
- Children Act 1989
- Adoption and Children Act 2002
- Family Law Act 1996
- Domestic Abuse Act 2021
- +17 more acts (22 total)

**Source 3: Form E Guidance** (To be added)
- Official Form E documentation
- Field-by-field explanations
- Asset valuation guides

**Source 4: Practice Directions** (Optional)
- Family Procedure Rules 2010
- Court protocols

---

## 🔧 Recommended Tech Stack

### **Option 1: Simple & Fast (MVP)**

**Vector Database: Chroma**
- Install: `pip install chromadb`
- Best for: <100K documents
- Storage: Local persistent
- Why: Easy to use, Python-native

**Embedding Model: bge-large-en-v1.5**
- Size: 1.3 GB
- Speed: 1000 docs/sec
- Quality: State-of-the-art for English
- Why: Best for legal text

**Dependencies:**
```bash
pip install chromadb==0.4.22
pip install sentence-transformers==2.2.2
pip install langchain==0.1.0
```

### **Option 2: Production (Scale)**

**Vector Database: Qdrant or Weaviate**
- Best for: Millions of documents
- Features: Advanced filtering, cloud-ready

---

## 📋 Chunking Strategy (CRITICAL!)

### **For Judgments: Paragraph-Level Chunks**

**Why paragraph-level?**
- Preserves legal structure
- Matches your fine-tuned model (trained on paragraph citations)
- Easy to cite in responses
- Optimal chunk size (500-1000 tokens)

**Example Chunk:**
```json
{
  "chunk_id": "ewfc_45_2023_para_15_18",
  "citation": "[2023] EWFC 45",
  "case_name": "Smith v Jones",
  "paragraphs": "15-18",
  "case_type": "financial_remedies",
  "year": 2023,
  "court": "EWFC",
  "content": "[Paragraphs 15-18 full text...]",
  "embedding": [0.234, -0.891, ...]
}
```

**Metadata to Track:**
- `citation`: e.g., "[2023] EWFC 45"
- `case_name`: "Smith v Jones"
- `year`: 2023
- `court`: "EWFC" / "EWHC" / "EWCOP"
- `case_type`: "financial_remedies" / "child_arrangements" / etc
- `paragraph_range`: "15-18"
- `judge`: "Justice Smith"
- `keywords`: ["lump sum", "business valuation"]

### **For Statutes: Section-Level Chunks**

```json
{
  "chunk_id": "mca_1973_s25",
  "act": "Matrimonial Causes Act 1973",
  "section": "Section 25",
  "subsection": "25(2)(a)",
  "title": "Matters to which court is to have regard",
  "content": "[Full section text...]",
  "related_sections": ["s23", "s24"],
  "embedding": [...]
}
```

---

## 💾 Database Sizing Estimates

### **Judgment Vector Database:**
- Source: 4,611 XML files (413 MB text)
- Chunks: ~46,000 chunks (avg 10 per judgment)
- Embeddings: 46,000 × 4 KB = ~184 MB
- Full text storage: ~500 MB
- **Total: ~700 MB - 1 GB**

### **Statute Database:**
- 22 acts, ~500 sections
- Embeddings: ~2 MB
- Text: ~50 MB
- **Total: ~52 MB**

### **Combined: ~750 MB - 1.1 GB**
✅ **Fits easily on any machine!**

---

## 🎯 When to Use RAG vs Fine-tuned Model

### **Scenario 1: Specific Fact Extraction** (User provides judgment)
```
Query: "What financial order was made?"
Context: [Judgment excerpt provided by user]

Route: FINE-TUNED LLAMA ONLY
Why: Model is expert at extracting from provided text
RAG: Not needed
```

### **Scenario 2: Multi-Case Analysis**
```
Query: "What are typical lump sum orders in high-value cases?"
Context: None provided

Route: RAG → FINE-TUNED LLAMA
Process:
1. RAG retrieves 10 relevant high-value financial remedy cases
2. Fine-tuned Llama analyzes and summarizes patterns
3. Cites specific cases and amounts
```

### **Scenario 3: Form E Guidance**
```
Query: "How should I report my pension on Form E?"
Context: None

Route: RAG → BASE LLAMA (Not fine-tuned!)
Why: Fine-tuned model has NO Form E training
Process:
1. RAG retrieves Form E field guidance
2. Base Llama-3.1-8B-Instruct explains
```

### **Scenario 4: Statutory Questions**
```
Query: "What factors does the court consider under Section 25?"
Context: None

Route: RAG → FINE-TUNED LLAMA
Process:
1. RAG retrieves Section 25 MCA 1973 text
2. RAG retrieves judgments applying Section 25
3. Fine-tuned Llama explains with case examples
```

### **Scenario 5: Report Generation**
```
Query: "Generate a judgment summary report"
Context: [Judgment provided or retrieved]

Route: RAG + FINE-TUNED LLAMA (Hybrid)
Process:
1. RAG retrieves full judgment if not provided
2. RAG retrieves related precedents
3. Fine-tuned Llama extracts key facts (its strength)
4. Base Llama generates narrative report (not trained for this)
```

---

## 🔍 Powerful Metadata Filtering

```python
# Filter by case type and year
results = collection.query(
    query_texts=["lump sum orders"],
    n_results=10,
    where={
        "case_type": "financial_remedies",
        "year": {"$gte": 2020}  # Cases from 2020 onwards
    }
)

# Filter by court level
results = collection.query(
    query_texts=["care proceedings"],
    where={"court": {"$in": ["EWFC", "EWHC"]}}
)

# Filter by multiple criteria
results = collection.query(
    query_texts=["property division"],
    where={
        "case_type": "financial_remedies",
        "year": {"$gte": 2020},
        "keywords": {"$contains": "business"}
    }
)
```

**Use cases:**
- Find similar cases
- Precedent research
- Pattern analysis
- Client-specific research

---

## 💡 Key Implementation Considerations

### **1. Chunk at Paragraph Level**
- Matches your fine-tuned model's training
- Preserves legal structure
- Easy to cite

### **2. Rich Metadata is Critical**
- Enable powerful filtering
- Track case types, years, courts
- Link related cases

### **3. Hybrid Search (Vector + Keyword)**
- Vector: Semantic similarity
- BM25: Keyword matching
- Combine for best results

### **4. Reranking**
- Use cross-encoder after initial retrieval
- Improves relevance
- Small performance cost

### **5. Query Routing is Essential**
- Don't always use RAG
- Match tool to task
- Save compute and improve speed

---

## ⚡ Performance Expectations

### **Query Speed:**
- Vector search: 10-50ms (46K chunks)
- Embedding query: 20-30ms
- Total retrieval: 30-80ms

### **Generation:**
- Concise response (28 tokens): 100-200ms
- Longer response (200 tokens): 500ms-1s

### **End-to-end: 150ms-1.5s** (Fast!)

---

## 🚀 Deployment Roadmap

### **Phase 1: MVP (Month 1)**
- Chroma vector DB (local)
- Index 4,611 XML judgments
- Simple top-k retrieval
- Fine-tuned Llama for synthesis
- **Use for:** Basic judgment QA, similar case finding

### **Phase 2: Enhanced (Months 2-3)**
- Add reranking
- Hybrid search (vector + BM25)
- Add statutes to DB
- Add Form E guidance
- **Use for:** Multi-case analysis, Form E help, precedent research

### **Phase 3: Production (Months 3-6)**
- Migrate to Qdrant/Weaviate
- Add caching layer (Redis)
- Monitoring & analytics
- Query optimization
- **Use for:** Full AILES platform

---

## 📊 Example: Complete Workflow

**User Query:**
```
"What are typical financial orders in cases involving small businesses?"
```

**Step 1: RAG Retrieval (30ms)**
```python
context = retrieve_relevant_context(
    query="financial orders small business valuation",
    filters={"case_type": "financial_remedies"},
    top_k=10
)

# Retrieved:
# - [2023] EWFC 45: Husband's IT consultancy valued at £2M
# - [2022] EWHC 123: Wife's bakery, £500K lump sum
# - [2021] EWFC 78: Partnership share division
# + 7 more cases
```

**Step 2: Fine-tuned Llama Synthesis (500ms)**
```python
prompt = f"""Analyze these financial remedy cases involving small businesses:

{format_retrieved_cases(context)}

Provide a summary of typical orders and valuation approaches."""

response = fine_tuned_llama.generate(prompt)
```

**Response:**
```
Based on recent UK family law judgments, typical financial orders
in small business cases include:

1. Business Valuation:
   - In [2023] EWFC 45, the husband's IT consultancy was valued at
     £2M using earnings-based method (paragraph 15)
   - [2022] EWHC 123 applied asset-based valuation for the wife's
     bakery (£500K) due to tangible assets (paragraph 22)

2. Division Approaches:
   - Lump sum orders preserving business continuity (5/10 cases)
   - Partnership share transfers (3/10 cases)
   - Deferred payments tied to business performance (2/10 cases)

3. Key Factors (Section 25 MCA 1973):
   - Active vs passive business involvement
   - Business liquidity and marketability
   - Impact on ongoing operations
```

**Citations are automatic because:**
- RAG provides specific cases
- Fine-tuned model learned citation patterns
- Paragraph numbers extracted from chunks

---

## ✅ Summary: Key Takeaways

1. **YES - Use your 4,611 XML judgment files as the database**
   - Already structured, perfect for RAG
   - 413 MB is very manageable
   - Chunk at paragraph level (500-1000 tokens)

2. **Fine-tuned Llama + RAG is COMPLEMENTARY**
   - Fine-tuned: Expert at extraction & synthesis
   - RAG: Provides the right context
   - Together: Powerful combination

3. **Start Simple: Chroma + bge-large-en-v1.5**
   - Works for 46K chunks
   - Easy to implement
   - Upgrade later if needed

4. **Metadata filtering is your secret weapon**
   - Filter by case type, year, court
   - Find similar cases instantly
   - Essential for precedent research

5. **Query routing is critical**
   - Don't always use RAG
   - Match tool to task
   - Saves compute, improves speed

6. **Total DB size: ~1 GB**
   - Fits on any machine
   - Fast retrieval (<100ms)
   - Room to grow

---

## 📚 Next Steps

1. **Set up Chroma vector database**
2. **Process XML files into paragraph chunks**
3. **Generate embeddings with bge-large-en-v1.5**
4. **Implement retrieval function**
5. **Integrate with fine-tuned Llama**
6. **Add query routing logic**
7. **Test on sample queries**
8. **Iterate and improve**

Good luck building AILES! 🚀
