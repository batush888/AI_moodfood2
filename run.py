import uvicorn

if __name__ == "__main__":
    # Use the enhanced API with Phase 3 features
    uvicorn.run("api.enhanced_routes:app", host="0.0.0.0", port=8000, reload=False)