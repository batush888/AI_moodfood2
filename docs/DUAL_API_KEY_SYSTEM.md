# Dual API Key System

This document describes the dual API key system implemented in the AI Mood Food Recommender, which provides redundancy and automatic fallback between OpenRouter and DeepSeek API keys.

## Overview

The system now supports two API keys for LLM operations:
- **Primary**: OpenRouter API key (existing)
- **Fallback**: DeepSeek API key (new)

When the primary API key hits rate limits or fails, the system automatically switches to the fallback key, ensuring continuous operation.

## Configuration

### Environment Variables

Add both API keys to your `.env` file:

```bash
# OpenRouter API Key (Primary)
OPENROUTER_API_KEY=sk-or-v1-your-openrouter-key-here

# DeepSeek API Key (Fallback)
DEEPSEEK_API_KEY=sk-or-v1-your-deepseek-key-here
```

### API Key Sources

Both keys are from OpenRouter:
- **OpenRouter Key**: Your existing OpenRouter API key
- **DeepSeek Key**: The new DeepSeek R1 0528 key you provided

## How It Works

### 1. Initialization
- System checks for both API keys on startup
- If both are available, API key rotation is enabled
- If only one is available, single key mode is used

### 2. Automatic Fallback
When an API call fails (rate limit, network error, etc.):
1. System detects the failure
2. Automatically rotates to the fallback API key
3. Retries the request with the new key
4. Logs the rotation for monitoring

### 3. Smart Rotation
- Rotation only happens once per request cycle
- System tracks which keys have been tried
- Prevents infinite rotation loops
- Maintains request context

## Benefits

### 🚀 **Reliability**
- No more single point of failure
- Automatic recovery from API issues
- Continuous operation during outages

### 📈 **Performance**
- Reduced downtime from rate limits
- Better handling of API quotas
- Improved user experience

### 🔒 **Security**
- Both keys remain server-side only
- No exposure to frontend
- Secure proxy endpoints maintained

## Implementation Details

### LLM Validator Changes
- `primary_api_key`: OpenRouter API key
- `fallback_api_key`: DeepSeek API key
- `current_api_key`: Currently active key
- `api_key_rotation_enabled`: Whether rotation is available

### API Call Flow
1. **Primary Attempt**: Try with OpenRouter key
2. **Failure Detection**: Monitor for 429s and errors
3. **Key Rotation**: Switch to DeepSeek key
4. **Retry**: Attempt request with fallback key
5. **Success/Failure**: Log results and continue

### Logging and Monitoring
- API key rotation events are logged
- Performance metrics track both keys
- Circuit breaker protects against cascading failures

## Testing

### Test Both API Keys
```bash
python test_api_key.py
```

### Test Dual System
```bash
python test_dual_api_system.py
```

### Manual Testing
1. Set both API keys in `.env`
2. Start the server
3. Check logs for "dual API keys" initialization message
4. Make requests to trigger rotation

## Monitoring

### Status Endpoint
Check `/status` endpoint for API key availability:
```json
{
  "services": {
    "openrouter": "available",
    "deepseek": "available",
    "gaode": "available"
  }
}
```

### Log Messages
Look for these log patterns:
- `"LLM validator initialized with dual API keys"`
- `"Switched to fallback API key (DeepSeek)"`
- `"Switched back to primary API key (OpenRouter)"`

## Troubleshooting

### Common Issues

#### 1. No API Key Rotation
**Problem**: System only uses one key
**Solution**: Ensure both environment variables are set

#### 2. Rotation Not Working
**Problem**: Fallback key not being used
**Solution**: Check that both keys are valid and have quota

#### 3. Performance Degradation
**Problem**: Slower response times
**Solution**: Monitor which key is being used and check quotas

### Debug Commands
```bash
# Check environment variables
env | grep -E "(OPENROUTER|DEEPSEEK)_API_KEY

# Test individual keys
curl -H "Authorization: Bearer $OPENROUTER_API_KEY" \
     https://openrouter.ai/api/v1/chat/completions

# Check server logs
tail -f data/logs/app.log | grep -E "(API key|rotation|fallback)"
```

## Migration Guide

### From Single Key
1. Add `DEEPSEEK_API_KEY` to your `.env`
2. Restart the server
3. Verify dual key initialization in logs
4. Test fallback functionality

### From Environment Variables
1. Update your deployment configuration
2. Add the new secret to Kubernetes
3. Restart pods
4. Verify both keys are loaded

## Security Considerations

### Key Management
- Never commit API keys to version control
- Use environment variables or secrets
- Rotate keys regularly
- Monitor usage and quotas

### Access Control
- Both keys access the same OpenRouter service
- No additional security risks
- Same rate limiting and quota policies

## Future Enhancements

### Planned Features
- **Dynamic Key Loading**: Hot-reload API keys without restart
- **Key Performance Metrics**: Track success rates per key
- **Automatic Key Testing**: Validate keys on startup
- **Key Rotation Policies**: Configurable rotation strategies

### Integration Possibilities
- **Multiple Providers**: Support for other LLM providers
- **Load Balancing**: Distribute requests across keys
- **Cost Optimization**: Route requests based on pricing
- **Geographic Routing**: Use keys from different regions

## Support

For issues with the dual API key system:
1. Check the troubleshooting section above
2. Review server logs for rotation events
3. Test individual API keys manually
4. Verify environment variable configuration

## Changelog

### v1.0.0 (Current)
- ✅ Dual API key support
- ✅ Automatic fallback on failure
- ✅ Smart rotation logic
- ✅ Comprehensive logging
- ✅ Kubernetes deployment support
- ✅ Testing and validation tools
