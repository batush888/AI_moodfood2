#!/usr/bin/env python3
"""
Secure Proxy Routes for External API Calls
Keeps API keys server-side and proxies requests to external services
Includes rate limiting, token safety checks, and input validation
"""

import os
import logging
import aiohttp
import asyncio
import time
import json
from collections import defaultdict, deque
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List
import re

# Setup logging
logger = logging.getLogger(__name__)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("Loaded environment variables from .env file")
except ImportError:
    logger.warning("python-dotenv not installed, using system environment variables")
except Exception as e:
    logger.warning(f"Failed to load .env file: {e}")

# Create router
router = APIRouter(prefix="/api", tags=["proxy"])

# Environment variables
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GAODE_API_KEY = os.getenv("GAODE_API_KEY")

# API endpoints
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GAODE_GEOCODE_URL = "https://restapi.amap.com/v3/geocode/regeo"
GAODE_WEATHER_URL = "https://restapi.amap.com/v3/weather/weatherInfo"

# Rate limiting configuration
RATE_LIMIT_REQUESTS = 10  # Maximum requests per minute per IP
RATE_LIMIT_WINDOW = 60    # Time window in seconds

# Token and input limits
MAX_TOKENS = 2000
MAX_INPUT_LENGTH = 4000

# Rate limiting storage (in production, use Redis or similar)
rate_limit_store = defaultdict(lambda: deque(maxlen=RATE_LIMIT_REQUESTS))

class OpenRouterRequest(BaseModel):
    """Request model for OpenRouter API with validation"""
    model: str = Field(default="deepseek/deepseek-r1-0528:free", description="Model to use")
    messages: List[Dict[str, str]] = Field(..., description="List of messages")
    max_tokens: Optional[int] = Field(default=256, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(default=0.0, ge=0.0, le=2.0, description="Temperature for generation")
    stream: Optional[bool] = Field(default=False, description="Whether to stream the response")
    
    @validator('max_tokens')
    def validate_max_tokens(cls, v):
        if v is not None and v > MAX_TOKENS:
            raise ValueError(f"max_tokens too high. Limit is {MAX_TOKENS}.")
        return v
    
    @validator('messages')
    def validate_messages(cls, v):
        if not v:
            raise ValueError("Messages cannot be empty")
        
        # Check total input length
        total_length = sum(len(msg.get('content', '')) for msg in v)
        if total_length > MAX_INPUT_LENGTH:
            raise ValueError(f"Input too long. Maximum {MAX_INPUT_LENGTH} characters allowed.")
        
        return v

class GaodeGeocodeRequest(BaseModel):
    """Request model for Gaode Geocoding API"""
    location: str = Field(..., description="Location in 'lng,lat' format")
    key: Optional[str] = None  # Will be injected server-side

class GaodeWeatherRequest(BaseModel):
    """Request model for Gaode Weather API"""
    city: str = Field(..., description="City code")
    key: Optional[str] = None  # Will be injected server-side
    extensions: str = Field(default="base", description="Weather data extensions")

def check_rate_limit(client_ip: str) -> bool:
    """
    Check if the client IP has exceeded the rate limit
    
    Args:
        client_ip: Client's IP address
        
    Returns:
        bool: True if within rate limit, False if exceeded
    """
    now = time.time()
    client_requests = rate_limit_store[client_ip]
    
    # Remove old requests outside the time window
    while client_requests and now - client_requests[0] > RATE_LIMIT_WINDOW:
        client_requests.popleft()
    
    # Check if we're at the limit
    if len(client_requests) >= RATE_LIMIT_REQUESTS:
        return False
    
    # Add current request
    client_requests.append(now)
    return True

def get_client_ip(request: Request) -> str:
    """
    Get the client's IP address from the request
    
    Args:
        request: FastAPI request object
        
    Returns:
        str: Client IP address
    """
    # Try to get real IP from headers (for proxy setups)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Fallback to direct IP
    return request.client.host if request.client else "unknown"

@router.post("/openai")
async def openai_proxy(request: OpenRouterRequest, http_request: Request):
    """
    Proxy endpoint for OpenAI-compatible requests.
    This is an alias for the /chat endpoint to maintain compatibility.
    """
    return await proxy_openrouter(request, http_request)

@router.post("/chat")
async def proxy_openrouter(request: OpenRouterRequest, http_request: Request):
    """
    Secure proxy for OpenRouter API calls
    Injects API key server-side and forwards request with rate limiting
    """
    client_ip = get_client_ip(http_request)
    
    # Check rate limit
    if not check_rate_limit(client_ip):
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded. Please wait and try again.",
                "retry_after": RATE_LIMIT_WINDOW
            }
        )
    
    try:
        if not OPENROUTER_API_KEY:
            logger.error("OPENROUTER_API_KEY not configured")
            return JSONResponse(
                status_code=500,
                content={"error": "OpenRouter API key not configured"}
            )
        
        # Prepare request payload
        payload = {
            "model": request.model,
            "messages": request.messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": request.stream
        }
        
        # Headers with API key
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://ai-moodfood.com",  # Your domain
            "X-Title": "AI Mood Food Recommendation System"
        }
        
        logger.info(f"Proxying OpenRouter request: model={request.model}, messages={len(request.messages)}, max_tokens={request.max_tokens}, IP={client_ip}")
        
        # Make request to OpenRouter
        async with aiohttp.ClientSession() as session:
            async with session.post(
                OPENROUTER_URL,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response_data = await response.json()
                
                if response.status != 200:
                    logger.error(f"OpenRouter API error: {response.status} - {response_data}")
                    return JSONResponse(
                        status_code=response.status,
                        content={
                            "error": f"OpenRouter API error: {response_data.get('error', {}).get('message', 'Unknown error')}",
                            "status_code": response.status
                        }
                    )
                
                logger.info(f"OpenRouter request successful: {len(response_data.get('choices', []))} choices, IP={client_ip}")
                return response_data
                
    except asyncio.TimeoutError:
        logger.error(f"OpenRouter API request timeout for IP: {client_ip}")
        return JSONResponse(
            status_code=504,
            content={"error": "OpenRouter API request timeout"}
        )
    except Exception as e:
        logger.error(f"Error proxying OpenRouter request for IP {client_ip}: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {str(e)}"}
        )

@router.get("/geocode")
async def proxy_gaode_geocode(location: str, http_request: Request):
    """
    Secure proxy for Gaode Geocoding API
    Injects API key server-side and forwards request with rate limiting
    """
    client_ip = get_client_ip(http_request)
    
    # Check rate limit
    if not check_rate_limit(client_ip):
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded. Please wait and try again.",
                "retry_after": RATE_LIMIT_WINDOW
            }
        )
    
    try:
        if not GAODE_API_KEY:
            logger.error("GAODE_API_KEY not configured")
            return JSONResponse(
                status_code=500,
                content={"error": "Gaode API key not configured"}
            )
        
        # Validate location format
        if not re.match(r'^-?\d+\.?\d*,-?\d+\.?\d*$', location):
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid location format. Expected 'lng,lat'"}
            )
        
        # Prepare request parameters
        params = {
            "location": location,
            "key": GAODE_API_KEY
        }
        
        logger.info(f"Proxying Gaode geocoding request: location={location}, IP={client_ip}")
        
        # Make request to Gaode
        async with aiohttp.ClientSession() as session:
            async with session.get(
                GAODE_GEOCODE_URL,
                params=params,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                response_data = await response.json()
                
                if response.status != 200:
                    logger.error(f"Gaode geocoding API error: {response.status} - {response_data}")
                    return JSONResponse(
                        status_code=response.status,
                        content={
                            "error": f"Gaode geocoding API error: {response_data.get('info', 'Unknown error')}",
                            "status_code": response.status
                        }
                    )
                
                logger.info(f"Gaode geocoding request successful, IP={client_ip}")
                return response_data
                
    except asyncio.TimeoutError:
        logger.error(f"Gaode geocoding API request timeout for IP: {client_ip}")
        return JSONResponse(
            status_code=504,
            content={"error": "Gaode geocoding API request timeout"}
        )
    except Exception as e:
        logger.error(f"Error proxying Gaode geocoding request for IP {client_ip}: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {str(e)}"}
        )

@router.get("/weather")
async def proxy_gaode_weather(city: str, http_request: Request, extensions: str = "base"):
    """
    Secure proxy for Gaode Weather API
    Injects API key server-side and forwards request with rate limiting
    """
    client_ip = get_client_ip(http_request)
    
    # Check rate limit
    if not check_rate_limit(client_ip):
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded. Please wait and try again.",
                "retry_after": RATE_LIMIT_WINDOW
            }
        )
    
    try:
        if not GAODE_API_KEY:
            logger.error("GAODE_API_KEY not configured")
            return JSONResponse(
                status_code=500,
                content={"error": "Gaode API key not configured"}
            )
        
        # Validate city parameter
        if not city or len(city) > 20:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid city parameter"}
            )
        
        # Validate extensions parameter
        if extensions not in ["base", "all"]:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid extensions parameter. Must be 'base' or 'all'"}
            )
        
        # Prepare request parameters
        params = {
            "city": city,
            "key": GAODE_API_KEY,
            "extensions": extensions
        }
        
        logger.info(f"Proxying Gaode weather request: city={city}, extensions={extensions}, IP={client_ip}")
        
        # Make request to Gaode
        async with aiohttp.ClientSession() as session:
            async with session.get(
                GAODE_WEATHER_URL,
                params=params,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                response_data = await response.json()
                
                if response.status != 200:
                    logger.error(f"Gaode weather API error: {response.status} - {response_data}")
                    return JSONResponse(
                        status_code=response.status,
                        content={
                            "error": f"Gaode weather API error: {response_data.get('info', 'Unknown error')}",
                            "status_code": response.status
                        }
                    )
                
                logger.info(f"Gaode weather request successful, IP={client_ip}")
                return response_data
                
    except asyncio.TimeoutError:
        logger.error(f"Gaode weather API request timeout for IP: {client_ip}")
        return JSONResponse(
            status_code=504,
            content={"error": "Gaode weather API request timeout"}
        )
    except Exception as e:
        logger.error(f"Error proxying Gaode weather request for IP {client_ip}: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {str(e)}"}
        )

@router.get("/health")
async def proxy_health():
    """Health check endpoint for proxy services"""
    return {
        "status": "healthy",
        "services": {
            "openrouter": "available" if OPENROUTER_API_KEY else "not_configured",
        "deepseek": "available" if os.getenv("DEEPSEEK_API_KEY") else "not_configured",
            "gaode": "available" if GAODE_API_KEY else "not_configured"
        },
        "rate_limiting": {
            "requests_per_minute": RATE_LIMIT_REQUESTS,
            "window_seconds": RATE_LIMIT_WINDOW
        },
        "limits": {
            "max_tokens": MAX_TOKENS,
            "max_input_length": MAX_INPUT_LENGTH
        },
        "timestamp": "2025-08-24T16:00:00Z"
    }

@router.get("/rate-limit-status")
async def rate_limit_status(http_request: Request):
    """Get current rate limit status for the client IP"""
    client_ip = get_client_ip(http_request)
    
    # Check rate limit
    if not check_rate_limit(client_ip):
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded. Please wait and try again.",
                "retry_after": RATE_LIMIT_WINDOW
            }
        )
    
    now = time.time()
    client_requests = rate_limit_store[client_ip]
    
    # Remove old requests outside the time window
    while client_requests and now - client_requests[0] > RATE_LIMIT_WINDOW:
        client_requests.popleft()
    
    return {
        "ip": client_ip,
        "requests_used": len(client_requests),
        "requests_limit": RATE_LIMIT_REQUESTS,
        "window_seconds": RATE_LIMIT_WINDOW,
        "reset_time": now + RATE_LIMIT_WINDOW if client_requests else now
    }
