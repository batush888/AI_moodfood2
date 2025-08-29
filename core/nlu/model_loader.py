#!/usr/bin/env python3
"""
Robust Model Loader with Hot-Reloading Support

This module provides centralized model loading logic with atomic symlink support
and thread-safe hot-reloading capabilities.
"""

import os
import threading
import joblib
import logging
import json
from pathlib import Path
from typing import Tuple, Any, Optional, Dict

logger = logging.getLogger(__name__)

# Constants
MODEL_SYMLINK = Path("models/intent_classifier/current")
VERSIONS_DIR = Path("models/intent_classifier/versions")
VERSIONS_INDEX = VERSIONS_DIR / "versions.json"

# Thread-safe lock for model operations
_lock = threading.RLock()

# In-memory cache for loaded models
_loaded = {
    "model": None,
    "vectorizer": None,
    "labels": [],
    "version": None,
    "metadata": {},
    "last_loaded": None
}

def _load_from_path(path: Path) -> Dict[str, Any]:
    """
    Load model data from a version directory.
    
    Args:
        path: Path to version directory
        
    Returns:
        Dict containing model data
        
    Raises:
        FileNotFoundError: If model files don't exist
        Exception: If loading fails
    """
    model_file = path / "model.joblib"
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    try:
        model_data = joblib.load(model_file)
        logger.info(f"Model loaded from {path}")
        return model_data
    except Exception as e:
        logger.error(f"Failed to load model from {path}: {e}")
        raise

def _load_metadata(path: Path) -> Dict[str, Any]:
    """
    Load metadata from version directory.
    
    Args:
        path: Path to version directory
        
    Returns:
        Dict containing metadata
    """
    metadata = {}
    
    # Try to load metrics.json
    metrics_file = path / "metrics.json"
    if metrics_file.exists():
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                metadata["metrics"] = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load metrics from {metrics_file}: {e}")
    
    # Try to load label_mappings.json
    labels_file = path / "label_mappings.json"
    if labels_file.exists():
        try:
            with open(labels_file, "r", encoding="utf-8") as f:
                metadata["label_mappings"] = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load label mappings from {labels_file}: {e}")
    
    return metadata

def reload_current_model() -> bool:
    """
    Force reload model from current symlink path into memory.
    
    Returns:
        bool: True if reload successful, False otherwise
    """
    with _lock:
        try:
            if not MODEL_SYMLINK.exists():
                logger.warning("Model symlink not present")
                _loaded.update({
                    "model": None, 
                    "vectorizer": None, 
                    "labels": [], 
                    "version": None,
                    "metadata": {},
                    "last_loaded": None
                })
                return False
            
            # Resolve symlink target
            if MODEL_SYMLINK.is_symlink():
                target = MODEL_SYMLINK.resolve()
            else:
                target = MODEL_SYMLINK
            
            if not target.exists():
                logger.error(f"Symlink target does not exist: {target}")
                return False
            
            # Load model data
            model_data = _load_from_path(target)
            
            # Load metadata
            metadata = _load_metadata(target)
            
            # Update cache
            _loaded.update({
                "model": model_data.get("classifier"),
                "vectorizer": model_data.get("vectorizer"),
                "labels": model_data.get("labels", []),
                "version": target.name,
                "metadata": metadata,
                "last_loaded": os.path.getmtime(target)
            })
            
            logger.info(f"Model loaded into memory (version={_loaded['version']})")
            return True
            
        except Exception as e:
            logger.exception("Failed to reload model")
            return False

def get_model() -> Tuple[Any, Any, list, Optional[str]]:
    """
    Get the currently loaded model components.
    
    Returns:
        Tuple of (model, vectorizer, labels, version)
    """
    with _lock:
        if _loaded["model"] is None:
            reload_current_model()
        return (
            _loaded["model"], 
            _loaded["vectorizer"], 
            _loaded["labels"], 
            _loaded["version"]
        )

def get_model_with_metadata() -> Tuple[Any, Any, list, Optional[str], Dict[str, Any]]:
    """
    Get the currently loaded model components with metadata.
    
    Returns:
        Tuple of (model, vectorizer, labels, version, metadata)
    """
    with _lock:
        if _loaded["model"] is None:
            reload_current_model()
        return (
            _loaded["model"], 
            _loaded["vectorizer"], 
            _loaded["labels"], 
            _loaded["version"],
            _loaded["metadata"]
        )

def get_current_version() -> Optional[str]:
    """
    Get the current model version without loading the full model.
    
    Returns:
        str: Current version ID or None if not available
    """
    with _lock:
        if _loaded["version"] is None:
            # Try to reload just to get version info
            try:
                if MODEL_SYMLINK.exists() and MODEL_SYMLINK.is_symlink():
                    target = MODEL_SYMLINK.resolve()
                    if target.exists():
                        _loaded["version"] = target.name
            except Exception:
                pass
        return _loaded["version"]

def is_model_loaded() -> bool:
    """
    Check if a model is currently loaded in memory.
    
    Returns:
        bool: True if model is loaded, False otherwise
    """
    with _lock:
        return _loaded["model"] is not None

def get_model_info() -> Dict[str, Any]:
    """
    Get information about the currently loaded model.
    
    Returns:
        Dict containing model information
    """
    with _lock:
        return {
            "version": _loaded["version"],
            "labels_count": len(_loaded["labels"]),
            "last_loaded": _loaded["last_loaded"],
            "metadata": _loaded["metadata"],
            "is_loaded": _loaded["model"] is not None
        }

def clear_cache():
    """
    Clear the in-memory model cache.
    Useful for testing or when you want to force a fresh load.
    """
    with _lock:
        _loaded.update({
            "model": None,
            "vectorizer": None,
            "labels": [],
            "version": None,
            "metadata": {},
            "last_loaded": None
        })
        logger.info("Model cache cleared")

def validate_model_integrity() -> bool:
    """
    Validate that the current model files are intact and loadable.
    
    Returns:
        bool: True if model is valid, False otherwise
    """
    try:
        if not MODEL_SYMLINK.exists():
            logger.warning("Model symlink does not exist")
            return False
        
        # Resolve symlink
        if MODEL_SYMLINK.is_symlink():
            target = MODEL_SYMLINK.resolve()
        else:
            target = MODEL_SYMLINK
        
        if not target.exists():
            logger.error(f"Symlink target does not exist: {target}")
            return False
        
        # Check required files
        required_files = ["model.joblib"]
        for file_name in required_files:
            file_path = target / file_name
            if not file_path.exists():
                logger.error(f"Required file missing: {file_path}")
                return False
        
        # Try to load model (without keeping in memory)
        try:
            model_data = _load_from_path(target)
            required_keys = ["classifier", "vectorizer", "labels"]
            for key in required_keys:
                if key not in model_data:
                    logger.error(f"Required key missing from model data: {key}")
                    return False
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False
        
        logger.info(f"Model validation passed for version: {target.name}")
        return True
        
    except Exception as e:
        logger.exception("Model validation failed with exception")
        return False

def list_available_versions() -> list:
    """
    List all available model versions.
    
    Returns:
        list: List of version directories
    """
    try:
        if not VERSIONS_DIR.exists():
            return []
        
        versions = []
        for version_dir in VERSIONS_DIR.iterdir():
            if version_dir.is_dir():
                versions.append({
                    "version": version_dir.name,
                    "path": str(version_dir),
                    "exists": version_dir.exists(),
                    "is_current": MODEL_SYMLINK.exists() and MODEL_SYMLINK.resolve() == version_dir
                })
        
        return sorted(versions, key=lambda x: x["version"], reverse=True)
        
    except Exception as e:
        logger.error(f"Failed to list versions: {e}")
        return []

# Initialize model on module import
def _initialize():
    """Initialize the model loader on module import."""
    try:
        # Ensure directories exist
        VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Try to load initial model
        if MODEL_SYMLINK.exists():
            reload_current_model()
        else:
            logger.info("No current model symlink found, will load on first request")
            
    except Exception as e:
        logger.warning(f"Model loader initialization failed: {e}")

# Auto-initialize
_initialize()
