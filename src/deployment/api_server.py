#!/usr/bin/env python3
"""
AILES Legal AI API Server - Complete Implementation
FastAPI server for ReGoBs integration with all three AI components
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import traceback

# Set up comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Request/Response Models
class PredictionRequest(BaseModel):
    instruction: str = Field(..., description="System instruction for the AI model")
    input: str = Field(..., description="User input text or JSON data")
    max_length: Optional[int] = Field(2048, description="Maximum response length")
    temperature: Optional[float] = Field(0.7, description="Generation temperature")

class PredictionResponse(BaseModel):
    response: str = Field(..., description="AI-generated response")
    metadata: Dict[str, Any] = Field(..., description="Response metadata")
    processing_time: float = Field(..., description="Processing time in seconds")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health status")
    component: str = Field(..., description="AI component type")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    memory_info: Dict[str, Any] = Field(..., description="Memory usage information")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Error type")
    timestamp: str = Field(..., description="Error timestamp")

# AI Model Wrapper Class
class LegalAIModel:
    def __init__(self, model_path: str, component: str):
        self.component = component
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model components
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        self.load_time = None
        
        # Load the model
        self._load_model()

    def _load_model(self):
        """Load the fine-tuned LoRA model with comprehensive error handling"""
        start_time = time.time()
        
        try:
            logger.info(f"🚀 Loading {self.component} model from {self.model_path}")
            
            # Validate model path exists
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model directory not found: {self.model_path}")
            
            # Check for required files
            required_files = ["adapter_config.json", "adapter_model.safetensors"]
            missing_files = []
            for file in required_files:
                if not (self.model_path / file).exists():
                    missing_files.append(file)
            
            if missing_files:
                raise FileNotFoundError(f"Missing required files: {missing_files}")
            
            # Load tokenizer
            logger.info("📚 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Ensure pad token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Load base model path from adapter config
            with open(self.model_path / "adapter_config.json", 'r') as f:
                adapter_config = json.load(f)
            
            base_model_path = "/users/bgxp240/ailes_legal_ai/models/base_models/llama-3.2-3b-instruct"
            
            # Load base model
            logger.info(f"🤖 Loading base model from {base_model_path}...")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                local_files_only=True
            )
            
            # Load LoRA adapter
            logger.info("🔧 Loading LoRA adapter...")
            self.model = PeftModel.from_pretrained(
                base_model,
                str(self.model_path),
                torch_dtype=torch.bfloat16
            )
            
            # Set to evaluation mode
            self.model.eval()
            
            # Mark as loaded
            self.is_loaded = True
            self.load_time = time.time() - start_time
            
            logger.info(f"✅ {self.component} model loaded successfully in {self.load_time:.2f}s")
            logger.info(f"🎯 Device: {self.device}")
            logger.info(f"📊 Model parameters: {self.model.num_parameters():,}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load {self.component} model: {str(e)}")
            logger.error(f"📋 Traceback: {traceback.format_exc()}")
            self.is_loaded = False
            raise

    def generate_response(self, instruction: str, input_text: str, max_length: int = 2048, temperature: float = 0.7) -> Dict[str, Any]:
        """Generate response from the fine-tuned model"""
        if not self.is_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        start_time = time.time()
        
        try:
            # Format input using Llama-3.2 chat template
            formatted_input = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            
            # Tokenize input
            inputs = self.tokenizer(
                formatted_input,
                return_tensors="pt",
                truncation=True,
                max_length=max_length // 2,
                padding=False
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract assistant response
            assistant_marker = "<|start_header_id|>assistant<|end_header_id|>\n\n"
            if assistant_marker in generated_text:
                response = generated_text.split(assistant_marker)[-1].strip()
            else:
                # Fallback extraction
                response = generated_text[len(formatted_input):].strip()
            
            # Clean up response
            if "<|eot_id|>" in response:
                response = response.split("<|eot_id|>")[0].strip()
            
            # Try to parse as JSON for structured responses
            try:
                parsed_response = json.loads(response)
                is_structured = True
                final_response = parsed_response
            except json.JSONDecodeError:
                is_structured = False
                final_response = response
            
            processing_time = time.time() - start_time
            
            return {
                "response": final_response,
                "metadata": {
                    "component": self.component,
                    "is_structured": is_structured,
                    "input_length": len(formatted_input),
                    "output_length": len(response),
                    "processing_time": processing_time,
                    "model_parameters": self.model.num_parameters(),
                    "device": str(self.device)
                },
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"❌ Generation error for {self.component}: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Generation failed: {str(e)}"
            )

    def get_memory_info(self) -> Dict[str, Any]:
        """Get current memory usage information"""
        memory_info = {"cpu_memory_mb": 0, "gpu_memory_mb": 0, "gpu_memory_allocated_mb": 0}
        
        try:
            import psutil
            memory_info["cpu_memory_mb"] = psutil.Process().memory_info().rss / 1024 / 1024
        except:
            pass
        
        if torch.cuda.is_available():
            try:
                memory_info["gpu_memory_mb"] = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
                memory_info["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
            except:
                pass
        
        return memory_info

# Initialize FastAPI application
app = FastAPI(
    title="AILES Legal AI API",
    description="AI-powered legal assistance API for family law cases - ReGoBs Integration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for ReGoBs integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Global model instance
model_instance: Optional[LegalAIModel] = None

# Custom exception handler
@app.exception_handler(Exception)
async def custom_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "error_type": type(exc).__name__,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    )

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global model_instance
    
    # Get configuration from environment variables
    model_path = os.getenv("MODEL_PATH")
    component = os.getenv("COMPONENT", "chatbot")
    
    logger.info(f"🚀 Starting AILES Legal AI API Server")
    logger.info(f"📦 Component: {component}")
    logger.info(f"📁 Model Path: {model_path}")
    
    if not model_path:
        logger.error("❌ MODEL_PATH environment variable not set")
        return
    
    try:
        model_instance = LegalAIModel(model_path, component)
        logger.info(f"✅ API server ready for {component} component")
    except Exception as e:
        logger.error(f"❌ Failed to initialize model: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global model_instance
    if model_instance:
        logger.info("🧹 Cleaning up model resources...")
        del model_instance
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check endpoint"""
    global model_instance
    
    memory_info = {}
    if model_instance:
        memory_info = model_instance.get_memory_info()
    
    return HealthResponse(
        status="healthy" if model_instance and model_instance.is_loaded else "unhealthy",
        component=model_instance.component if model_instance else "unknown",
        model_loaded=model_instance.is_loaded if model_instance else False,
        gpu_available=torch.cuda.is_available(),
        memory_info=memory_info
    )

# Main prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Main prediction endpoint for any component"""
    global model_instance
    
    if not model_instance or not model_instance.is_loaded:
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        result = model_instance.generate_response(
            instruction=request.instruction,
            input_text=request.input,
            max_length=request.max_length,
            temperature=request.temperature
        )
        
        return PredictionResponse(
            response=json.dumps(result["response"]) if isinstance(result["response"], dict) else str(result["response"]),
            metadata=result["metadata"],
            processing_time=result["processing_time"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Component-specific endpoints for ReGoBs integration
@app.post("/chatbot/predict", response_model=PredictionResponse)
async def chatbot_predict(request: PredictionRequest):
    """Chatbot-specific prediction endpoint"""
    if model_instance and model_instance.component != "chatbot":
        raise HTTPException(status_code=400, detail="This endpoint requires chatbot model")
    
    return await predict(request)

@app.post("/predictor/predict", response_model=PredictionResponse)
async def predictor_predict(request: PredictionRequest):
    """Predictor-specific prediction endpoint"""
    if model_instance and model_instance.component != "predictor":
        raise HTTPException(status_code=400, detail="This endpoint requires predictor model")
    
    return await predict(request)

@app.post("/explainer/predict", response_model=PredictionResponse)
async def explainer_predict(request: PredictionRequest):
    """Explainer-specific prediction endpoint"""
    if model_instance and model_instance.component != "explainer":
        raise HTTPException(status_code=400, detail="This endpoint requires explainer model")
    
    return await predict(request)

# Information endpoints
@app.get("/info")
async def get_info():
    """Get API and model information"""
    global model_instance
    
    info = {
        "api_version": "1.0.0",
        "service": "AILES Legal AI",
        "component": model_instance.component if model_instance else "unknown",
        "model_loaded": model_instance.is_loaded if model_instance else False,
        "gpu_available": torch.cuda.is_available(),
        "endpoints": [
            "/health",
            "/predict", 
            "/chatbot/predict",
            "/predictor/predict",
            "/explainer/predict",
            "/info"
        ]
    }
    
    if model_instance and model_instance.is_loaded:
        info.update({
            "model_parameters": model_instance.model.num_parameters(),
            "load_time": model_instance.load_time,
            "device": str(model_instance.device)
        })
    
    return info

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AILES Legal AI API - ReGoBs Integration",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    # Get configuration from environment
    port = int(os.getenv("PORT", 8080))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"🚀 Starting AILES Legal AI API Server on {host}:{port}")
    
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=False,
        workers=1,
        log_level="info"
    )
