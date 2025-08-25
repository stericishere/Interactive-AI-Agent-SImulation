# Dating Show Frontend Service - Production Deployment Guide

## 🚀 Epic 4: Production Deployment & Validation

This guide provides comprehensive deployment instructions for the Dating Show Frontend Service with unified architecture integration.

## 📋 Pre-Deployment Checklist

### System Requirements
- **Python**: 3.9+
- **Node.js**: 16+ (for development tools)
- **Memory**: 2GB+ RAM recommended
- **Disk**: 5GB+ available space
- **Network**: Outbound access to backend service

### Required Dependencies
```bash
# Core dependencies
pip install fastapi uvicorn[standard] websockets
pip install jinja2 aiofiles python-multipart
pip install pydantic-settings python-dotenv
pip install aiohttp psutil  # For health checks

# Optional monitoring
pip install prometheus-client structlog
```

## 🏗️ Architecture Overview

```mermaid
graph TB
    LB[Load Balancer] --> FS[Frontend Service :8001]
    FS --> WS[WebSocket Router]
    FS --> UA[Unified Architecture]
    FS --> BS[Backend Service :8001]
    
    WS --> CM[Connection Manager]
    UA --> UAM[UnifiedAgentManager]
    UA --> UP[UpdatePipeline]
    UA --> FSA[FrontendStateAdapter]
    
    FS --> ST[Static Files]
    FS --> TP[Templates]
    ST --> JS[JavaScript Components]
```

## 🔧 Environment Configuration

### 1. Environment Variables (.env)

Create a `.env` file in the service root:

```bash
# Environment
DATING_SHOW_ENV=production
SECRET_KEY=your-secure-secret-key-here
PORT=8001

# Backend Integration
DATING_SHOW_BACKEND_URL=http://localhost:8001
SIMULATION_DATA_PATH=/data/simulation

# Performance
MAX_CONNECTIONS=500
CONNECTION_TIMEOUT=60
LOG_LEVEL=INFO
LOG_FILE=logs/dating_show_frontend.log

# CORS (Production)
ALLOWED_ORIGINS=https://your-domain.com,https://api.your-domain.com

# Optional: Database (if needed)
DATABASE_URL=postgresql://user:pass@localhost/dating_show
```

### 2. Production Settings Validation

Verify configuration:
```bash
python -c "from core.config import Settings; s = Settings(); print(f'Environment: {s.environment}'); print(f'Debug: {s.debug}'); print(f'Host: {s.host}'); print(f'CORS: {s.allowed_origins}')"
```

## 📦 Deployment Methods

### Method 1: Direct ASGI Deployment (Recommended)

#### Using Uvicorn with Production Config
```bash
# Install production dependencies
pip install uvloop httptools

# Start service with production ASGI config
python asgi_production.py

# Or using uvicorn directly
uvicorn asgi_production:application \
  --host 0.0.0.0 \
  --port 8001 \
  --workers 1 \
  --loop uvloop \
  --http httptools \
  --ws websockets \
  --log-level info \
  --access-log
```

#### Using Gunicorn + Uvicorn Workers
```bash
# Install gunicorn
pip install gunicorn

# Start with gunicorn
gunicorn asgi_production:application \
  -w 1 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8001 \
  --log-level info \
  --access-logfile - \
  --error-logfile -
```

### Method 2: Docker Deployment

#### Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create logs directory
RUN mkdir -p logs

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

# Start application
CMD ["python", "asgi_production.py"]
```

#### Docker Compose
```yaml
version: '3.8'

services:
  frontend:
    build: .
    ports:
      - "8001:8001"
    environment:
      - DATING_SHOW_ENV=production
      - DATING_SHOW_BACKEND_URL=http://backend:8001
    volumes:
      - ./logs:/app/logs
      - ./data:/data
    depends_on:
      - backend
    restart: unless-stopped
    networks:
      - dating-show

  backend:
    # Backend service configuration
    ports:
      - "8001:8001"
    networks:
      - dating-show

networks:
  dating-show:
    driver: bridge
```

### Method 3: Systemd Service

#### Service File (/etc/systemd/system/dating-show-frontend.service)
```ini
[Unit]
Description=Dating Show Frontend Service
After=network.target

[Service]
Type=exec
User=dating-show
Group=dating-show
WorkingDirectory=/opt/dating-show/frontend_service
Environment=PATH=/opt/dating-show/venv/bin
Environment=DATING_SHOW_ENV=production
ExecStart=/opt/dating-show/venv/bin/python asgi_production.py
ExecReload=/bin/kill -HUP $MAINPID
KillMode=mixed
TimeoutStopSec=5
PrivateTmp=true
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### Service Management
```bash
# Enable and start service
sudo systemctl enable dating-show-frontend
sudo systemctl start dating-show-frontend

# Check status
sudo systemctl status dating-show-frontend

# View logs
sudo journalctl -u dating-show-frontend -f
```

## 🔍 Health Monitoring

### Health Check Endpoints

#### System Health
```bash
curl http://localhost:8001/health
```

Response:
```json
{
  "overall_status": "healthy",
  "timestamp": "2024-08-22T12:00:00Z",
  "service": "dating-show-frontend",
  "version": "1.0.0",
  "environment": "production",
  "summary": {
    "total_checks": 9,
    "healthy": 8,
    "degraded": 1,
    "unhealthy": 0,
    "success_rate": 88.89
  },
  "checks": {
    "database_connection": {"status": "healthy", "message": "No database configured"},
    "dating_show_backend": {"status": "healthy", "message": "Backend connection successful"},
    "unified_architecture": {"status": "healthy", "message": "Components available"},
    "websocket_support": {"status": "healthy", "message": "WebSocket operational"},
    "file_system": {"status": "healthy", "message": "File system access OK"},
    "memory_usage": {"status": "healthy", "message": "Memory usage normal: 45.2%"},
    "disk_space": {"status": "healthy", "message": "Disk space OK: 78.3% free"},
    "update_pipeline": {"status": "degraded", "message": "Pipeline available, fallback mode"},
    "circuit_breaker": {"status": "healthy", "message": "Circuit breaker closed"}
  }
}
```

#### Individual Health Checks
```bash
# Check specific component
curl http://localhost:8001/health/dating_show_backend
curl http://localhost:8001/health/websocket_support
curl http://localhost:8001/health/unified_architecture
```

### Monitoring Integration

#### Prometheus Metrics (Optional)
```python
# Add to main.py for Prometheus integration
from prometheus_client import Counter, Histogram, generate_latest

REQUEST_COUNT = Counter('requests_total', 'Total requests')
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

#### Log Monitoring
```bash
# Monitor application logs
tail -f logs/dating_show_frontend.log

# Monitor system logs
journalctl -u dating-show-frontend -f

# Search for errors
grep -i error logs/dating_show_frontend.log
```

## 🌐 Reverse Proxy Configuration

### Nginx Configuration
```nginx
upstream dating_show_frontend {
    server 127.0.0.1:8001;
}

server {
    listen 80;
    server_name your-domain.com;

    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    # SSL Configuration
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;

    # Static files
    location /static/ {
        alias /opt/dating-show/frontend_service/static/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }

    # WebSocket support
    location /api/ws/ {
        proxy_pass http://dating_show_frontend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }

    # Application
    location / {
        proxy_pass http://dating_show_frontend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Health check
    location /health {
        proxy_pass http://dating_show_frontend;
        access_log off;
    }
}
```

## 📊 Performance Tuning

### Uvicorn Optimization
```bash
# Production optimized startup
uvicorn asgi_production:application \
  --host 0.0.0.0 \
  --port 8001 \
  --workers 1 \
  --loop uvloop \
  --http httptools \
  --ws websockets \
  --log-level info \
  --no-access-log \
  --server-header false \
  --date-header false
```

### Resource Limits
```bash
# Set resource limits in systemd service
[Service]
LimitNOFILE=65536
LimitNPROC=4096
MemoryMax=2G
CPUQuota=200%
```

### WebSocket Optimization
```python
# In core/config.py
class Settings(BaseSettings):
    # WebSocket settings
    websocket_max_connections: int = 1000
    websocket_heartbeat_interval: int = 30
    websocket_max_message_size: int = 1024 * 1024  # 1MB
    websocket_timeout: int = 300  # 5 minutes
```

## 🚨 Troubleshooting

### Common Issues

#### 1. WebSocket Connection Failures
```bash
# Check WebSocket endpoint
curl -i -N -H "Connection: Upgrade" -H "Upgrade: websocket" \
  http://localhost:8001/api/ws/agents/general

# Verify proxy configuration
nginx -t
systemctl reload nginx
```

#### 2. Static Files Not Loading
```bash
# Check file permissions
ls -la static/
chmod -R 644 static/
chmod 755 static/

# Verify mount points in main.py
app.mount("/static", StaticFiles(directory="static_dirs"), name="static")
app.mount("/static/js", StaticFiles(directory="static/js"), name="static_js")
```

#### 3. Backend Connection Issues
```bash
# Test backend connectivity
curl http://localhost:8001/health

# Check backend URL configuration
echo $DATING_SHOW_BACKEND_URL

# Verify network connectivity
ping backend-host
telnet backend-host 8001
```

#### 4. Memory/Performance Issues
```bash
# Monitor resource usage
htop
free -h
df -h

# Check application performance
curl http://localhost:8001/health/memory_usage
curl http://localhost:8001/health/update_pipeline
```

### Debug Mode Activation
```bash
# Temporarily enable debug mode
export DATING_SHOW_ENV=development
export LOG_LEVEL=DEBUG

# Restart service
systemctl restart dating-show-frontend

# Monitor detailed logs
journalctl -u dating-show-frontend -f
```

## 🔐 Security Considerations

### Production Security Checklist
- [ ] Generate secure SECRET_KEY
- [ ] Configure proper CORS origins
- [ ] Enable SSL/TLS (HTTPS)
- [ ] Set up proper firewall rules
- [ ] Use non-root user for service
- [ ] Regular security updates
- [ ] Log monitoring and alerting
- [ ] Rate limiting (via reverse proxy)
- [ ] Input validation and sanitization

### Security Headers (Nginx)
```nginx
# Add security headers
add_header X-Content-Type-Options nosniff;
add_header X-Frame-Options DENY;
add_header X-XSS-Protection "1; mode=block";
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline' https://d3js.org; style-src 'self' 'unsafe-inline';";
```

## 📈 Scaling Considerations

### Horizontal Scaling
- Use multiple frontend service instances behind load balancer
- Implement session affinity for WebSocket connections
- Consider Redis for shared session storage
- Use CDN for static file delivery

### Vertical Scaling
- Monitor CPU/memory usage
- Adjust worker processes based on load
- Optimize database queries
- Implement caching strategies

## 🎯 Epic 4 Completion Checklist

- [x] **ASGI Server Configuration**: Uvicorn/Gunicorn production config
- [x] **WebSocket Integration**: FastAPI WebSocket router with unified architecture
- [x] **Frontend Integration**: JavaScript components integrated into templates
- [x] **Environment Configuration**: Production settings and environment variables
- [x] **Health Monitoring**: Comprehensive health checks with performance metrics
- [x] **Deployment Documentation**: Complete deployment guide and troubleshooting
- [ ] **End-to-End Validation**: Full system functionality testing

## 🔄 Next Steps

1. **Load Testing**: Test with simulated load to validate performance
2. **Monitoring Setup**: Implement comprehensive monitoring and alerting
3. **Backup Strategy**: Implement data backup and recovery procedures
4. **Documentation**: Create user documentation and API reference
5. **CI/CD Pipeline**: Automate deployment and testing processes

---

**Epic 4: Production Deployment & Validation** provides enterprise-grade deployment capabilities for the Dating Show unified architecture, ensuring reliable, scalable, and secure production operations.