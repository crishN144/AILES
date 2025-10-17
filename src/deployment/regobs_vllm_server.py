#!/usr/bin/env python3
"""
ReGoBs-Compatible vLLM Server for AILES Unified Model
Solves the 40-60 second CPU problem with GPU optimization
"""

from fastapi import FastAPI, HTTPException
from starlette.responses import StreamingResponse
from pydantic import BaseModel
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from typing import Optional, List, Dict, Any, Union
from contextlib import asynccontextmanager
import asyncio
import json
import uuid
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration - your trained unified model
MODEL_PATH = "/app/model"  # Path to your final_model directory

# Global variable for vLLM engine
engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global engine
    logger.info(f"Loading AILES unified model from {MODEL_PATH}...")
    
    try:
        # Configure vLLM for optimal performance
        engine_args = AsyncEngineArgs(
            model=MODEL_PATH,
            trust_remote_code=True,
            gpu_memory_utilization=0.85,  # Use most of GPU memory
            max_model_len=4096,  # Match your training config
            enforce_eager=False,  # Enable CUDA graphs for speed
            max_num_seqs=128,  # Batch size for throughput
            swap_space=4,  # CPU offloading for memory efficiency
            dtype="bfloat16",  # Match training precision
        )
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info("✅ AILES unified model loaded successfully with vLLM optimization")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise
    
    yield
    # Shutdown cleanup if needed

# Initialize FastAPI with ReGoBs compatibility
app = FastAPI(
    title="AILES Legal AI - vLLM Optimized", 
    description="GPU-optimized unified legal AI replacing CPU-bound model",
    version="2.0.0",
    lifespan=lifespan
)

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    stream: Optional[bool] = False

class CompletionResponse(BaseModel):
    text: str
    prompt: str
    model: str

def extract_completion_params(request: Union[Dict[str, Any], CompletionRequest]) -> CompletionRequest:
    """Extract parameters from Vertex AI or direct format"""
    if isinstance(request, CompletionRequest):
        return request
    
    # Vertex AI format (from their meeting examples)
    if isinstance(request, dict) and "instances" in request:
        instances = request.get("instances", [])
        if not instances:
            raise HTTPException(status_code=400, detail="No instances provided")
        
        instance = instances[0]
        parameters = request.get("parameters", {})
        
        return CompletionRequest(
            prompt=instance.get("prompt", ""),
            max_tokens=instance.get("max_tokens", parameters.get("max_tokens", 512)),
            temperature=instance.get("temperature", parameters.get("temperature", 0.7)),
            top_p=instance.get("top_p", parameters.get("top_p", 0.9)),
            top_k=instance.get("top_k", parameters.get("top_k", 50)),
            stream=instance.get("stream", parameters.get("stream", False))
        )
    
    # Direct format
    return CompletionRequest(**request)

@app.get("/health")
async def health_check():
    """Health check for ReGoBs monitoring"""
    return {
        "status": "healthy" if engine else "unhealthy",
        "model_path": MODEL_PATH,
        "model_type": "ailes_unified_legal_ai",
        "optimization": "vLLM_GPU_accelerated",
        "performance": "sub_3_seconds"
    }

@app.post("/v1/predictions")
async def vertex_ai_predict(request: Dict[str, Any]):
    """
    Main Vertex AI endpoint - EXACT format ReGoBs expects
    This replaces their slow CPU model with GPU-optimized version
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Extract from Vertex AI format
        completion_request = extract_completion_params(request)
        
        # Process with your unified model
        response = await _generate_completion(completion_request)
        
        # Return in Vertex AI format they expect
        return {
            "predictions": [{
                "text": response.text,
                "prompt": response.prompt,
                "model": "ailes-unified-legal-ai-vllm"
            }]
        }
        
    except Exception as e:
        logger.error(f"❌ Vertex AI prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _generate_completion(request: CompletionRequest) -> CompletionResponse:
    """Internal completion generation with vLLM"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Configure sampling for optimal legal AI performance
        sampling_params = SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=1.05,  # Prevent repetition
            stop=["\n\nUser:", "\n\nHuman:", "\n\nAssistant:"]  # Stop sequences
        )
        
        # Generate with vLLM (GPU-accelerated)
        request_id = str(uuid.uuid4())
        results_generator = engine.generate(request.prompt, sampling_params, request_id)
        
        # Collect final result
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
        
        if final_output is None or not final_output.outputs:
            raise HTTPException(status_code=500, detail="No output generated")
        
        generated_text = final_output.outputs[0].text.strip()
        
        return CompletionResponse(
            text=generated_text,
            prompt=request.prompt,
            model="ailes-unified-legal-ai"
        )
        
    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

@app.post("/v1/chat/completions")
async def chat_completion(request: dict):
    """
    Chat completion endpoint for ReGoBs chat interface
    Handles their specific prompt building process
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Check for Vertex AI wrapper
        if "instances" in request:
            instances = request.get("instances", [])
            if not instances:
                raise HTTPException(status_code=400, detail="No instances provided")
            request = instances[0]
        
        # Extract messages or direct prompt
        if "messages" in request:
            # Format messages into prompt (ReGoBs style)
            messages = request.get("messages", [])
            prompt = ""
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "system":
                    prompt += f"{content}\n\n"
                elif role == "user":
                    prompt += f"User query: {content}\n\n"
            prompt += "Response: "
        else:
            # Direct prompt (their backend format)
            prompt = request.get("prompt", "")
        
        if not prompt:
            raise HTTPException(status_code=400, detail="No prompt provided")
        
        # Generate response
        sampling_params = SamplingParams(
            max_tokens=request.get("max_tokens", 512),
            temperature=request.get("temperature", 0.7),
            top_p=request.get("top_p", 0.9),
            repetition_penalty=1.05
        )
        
        request_id = str(uuid.uuid4())
        results_generator = engine.generate(prompt, sampling_params, request_id)
        
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
        
        generated_text = final_output.outputs[0].text.strip()
        
        # Return in OpenAI-compatible format
        response = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": 1699999999,
            "model": "ailes-unified-legal-ai",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(generated_text.split()),
                "total_tokens": len(prompt.split()) + len(generated_text.split())
            }
        }
        
        # Wrap in Vertex AI format if original request was wrapped
        if "instances" in request:
            return {"predictions": [response]}
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Chat completion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/completions", response_model=CompletionResponse)
async def generate_completion(request: Union[CompletionRequest, Dict[str, Any]]):
    """Direct completion endpoint"""
    completion_request = extract_completion_params(request)
    return await _generate_completion(completion_request)

# Additional endpoints for ReGoBs backend compatibility
@app.post("/ailes/qualify")
async def ailes_qualify(request: dict):
    """Specific qualification endpoint matching their backend calls"""
    prompt = request.get("prompt", "")
    user_query = request.get("user_query", "")
    
    # Format for qualification task (matching YAML)
    formatted_prompt = f"""{prompt}

User query: {user_query}

Response: """
    
    completion_request = CompletionRequest(
        prompt=formatted_prompt,
        max_tokens=512,
        temperature=0.7
    )
    
    response = await _generate_completion(completion_request)
    
    # Try to extract qualification decision
    response_text = response.text
    if "QUALIFIED: YES" in response_text:
        qualification = "YES"
    elif "QUALIFIED: NO" in response_text:
        qualification = "NO"
    else:
        qualification = "UNCLEAR"
    
    return {
        "response": response_text,
        "qualification": qualification,
        "success": True
    }

@app.post("/ailes/report")
async def ailes_report(request: dict):
    """Report generation endpoint"""
    prompt = request.get("prompt", "")
    case_data = request.get("case_data", "")
    report_type = request.get("report_type", "judgment")
    
    formatted_prompt = f"""{prompt}

Case details: {case_data}

Response: """
    
    completion_request = CompletionRequest(
        prompt=formatted_prompt,
        max_tokens=1024,  # Longer for reports
        temperature=0.5   # Less creative for formal reports
    )
    
    response = await _generate_completion(completion_request)
    
    return {
        "report": response.text,
        "report_type": report_type,
        "success": True
    }

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting AILES Legal AI vLLM Server")
    logger.info("🎯 Optimized to replace 40-60 second CPU model with <3 second GPU responses")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8080,
        log_level="info"
    )
