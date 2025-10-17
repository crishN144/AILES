# AILES - Core Legal AI System

Essential components of the AILES legal AI system for UK family court judgment analysis.

## 📁 Structure

```
AILES/
├── src/
│   ├── data_processing/    # XML parsing and data cleaning
│   ├── training/          # Model training scripts (LLaMA, Mistral)
│   ├── validation/        # Testing and validation
│   └── deployment/        # API server
└── data/raw/xml_judgments/ # UK family court XML judgments
```

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Process XML judgments:**
   ```bash
   python src/data_processing/enhanced_xml_parser.py
   ```

3. **Run API server:**
   ```bash
   python src/deployment/api_server.py
   ```

## 🔬 Core Components

- **Data Processing:** Parse and clean UK family court XML documents
- **Training:** Fine-tune LLMs on legal data
- **Validation:** Test model performance and accuracy
- **Deployment:** Production API for legal document analysis

## 📊 Dataset

Contains XML files of UK family court judgments from EWHC (Family Division).

## 📄 License

MIT License
