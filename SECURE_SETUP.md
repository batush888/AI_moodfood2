# 🔒 Secure API Key Setup Guide

This guide explains how to set up the secure proxy system that keeps your API keys safe and never exposes them to the frontend.

## 🎯 **What This Solves**

- ✅ **API keys are never exposed** to the browser or GitHub
- ✅ **All external API calls** go through secure backend proxy
- ✅ **Environment variables** keep keys secure server-side
- ✅ **Deployment-ready** for platforms like Vercel, Render, Netlify
- ✅ **Same functionality** - app works exactly the same for users

## 📋 **Setup Steps**

### **1. Create .env File**

Copy the template and add your actual API keys:

```bash
# Copy the template
cp env_template.txt .env

# Edit .env with your actual API keys
nano .env
```

Your `.env` file should look like this:

```env
# API Keys - Keep these secure and never expose to frontend
OPENROUTER_API_KEY=sk-or-v1-your-actual-openrouter-key-here
DEEPSEEK_API_KEY=sk-or-v1-your-actual-deepseek-key-here
GAODE_API_KEY=your-actual-gaode-key-here

# Application Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1
```

### **2. Get Your API Keys**

#### **OpenRouter API Key:**
1. Go to [https://openrouter.ai/keys](https://openrouter.ai/keys)
2. Create an account and generate a new API key
3. Copy the key (starts with `sk-or-v1-`)

#### **Gaode/Amap API Key:**
1. Go to [https://lbs.amap.com/](https://lbs.amap.com/)
2. Create an account and apply for an API key
3. Copy the key

### **3. Start the Backend Server**

```bash
# Install dependencies (if not already done)
pip install -r requirements.txt

# Start the secure backend server
PYTHONPATH=. python api/enhanced_routes.py
```

The server will start on `http://localhost:8000` with secure proxy endpoints.

### **4. Test the Setup**

Run the security test to verify everything is working:

```bash
python test_secure_proxy.py
```

You should see:
```
🎉 All security tests passed! API keys are secure.
```

### **5. Start the Frontend**

In a new terminal:

```bash
# Start frontend server
python -m http.server 8080 --directory frontend

# Open in browser
open http://localhost:8080
```

## 🔧 **How It Works**

### **Before (Insecure):**
```
Frontend → OpenRouter API (with exposed key)
Frontend → Gaode API (with exposed key)
```

### **After (Secure):**
```
Frontend → Backend Proxy → OpenRouter API (key injected server-side)
Frontend → Backend Proxy → Gaode API (key injected server-side)
```

## 📡 **API Endpoints**

### **Secure Proxy Endpoints:**

- `POST /api/chat` - OpenRouter proxy
- `GET /api/geocode` - Gaode geocoding proxy  
- `GET /api/weather` - Gaode weather proxy
- `GET /api/health` - Health check

### **Example Usage:**

```javascript
// Frontend calls (no API keys needed)
fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        model: 'deepseek/deepseek-r1-0528:free',
        messages: [{ role: 'user', content: 'Hello' }]
    })
})

fetch('/api/geocode?location=116.4074,39.9042')
fetch('/api/weather?city=110101&extensions=base')
```

## 🚀 **Deployment**

### **Vercel/Netlify Functions:**

1. Set environment variables in your deployment platform
2. Deploy the backend code
3. Update frontend `API_BASE` to point to your deployed backend

### **Docker:**

```dockerfile
FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["python", "api/enhanced_routes.py"]
```

### **Traditional Hosting:**

1. Upload backend files to your server
2. Set environment variables
3. Run with `uvicorn api.enhanced_routes:app --host 0.0.0.0 --port 8000`

## 🔍 **Security Verification**

### **Check These Files:**

1. **frontend/config.js** - Should NOT contain API keys
2. **frontend/index.html** - Should use proxy endpoints
3. **.env** - Should contain your actual API keys
4. **.gitignore** - Should include `.env`

### **Run Security Tests:**

```bash
python test_secure_proxy.py
```

## 🛠️ **Troubleshooting**

### **API Key Not Working:**
- Check `.env` file exists and has correct keys
- Verify keys are valid and have proper permissions
- Check server logs for authentication errors

### **Proxy Endpoints Not Responding:**
- Ensure backend server is running
- Check CORS settings if calling from different domain
- Verify endpoint URLs in frontend config

### **Weather/Location Not Working:**
- Check Gaode API key permissions
- Verify geocoding and weather endpoints are enabled
- Test proxy endpoints directly with curl

## 📚 **Additional Resources**

- [OpenRouter API Documentation](https://openrouter.ai/docs)
- [Gaode/Amap API Documentation](https://lbs.amap.com/api)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Environment Variables Best Practices](https://12factor.net/config)

## ✅ **Success Checklist**

- [ ] `.env` file created with real API keys
- [ ] Backend server starts without errors
- [ ] `test_secure_proxy.py` passes all tests
- [ ] Frontend loads and works normally
- [ ] Weather detection works
- [ ] No API keys visible in browser dev tools
- [ ] Ready for deployment

---

**🎉 Congratulations!** Your app is now secure and ready for public deployment without exposing API keys.
