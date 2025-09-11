# Production Deployment Guide

## Quick Start with Docker

### 1. Environment Setup
```bash
# Copy and configure environment variables
cp .env.production.example .env.production
# Edit .env.production with your actual values
```

### 2. Deploy with Docker Compose
```bash
# Start the full production stack
docker-compose -f docker-compose.prod.yml --env-file .env.production up -d

# Check status
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs -f trading-bot
```

### 3. Access Services
- **Trading Bot Dashboard**: http://localhost:8080
- **Grafana Monitoring**: http://localhost:3000 (admin/your_grafana_password)
- **Prometheus Metrics**: http://localhost:9090
- **PostgreSQL**: localhost:5432

## Render.com Deployment

### Option 1: Docker Deployment (Recommended)
1. Connect your GitHub repo to Render
2. Create a **Web Service** with these settings:
   - **Build Command**: `docker build -f Dockerfile.prod -t trading-bot .`
   - **Start Command**: `docker run -p 10000:8080 trading-bot`
   - **Environment Variables**: Set all variables from `.env.production.example`

### Option 2: Native Python Deployment
1. **Build Command**: `pip install -r requirements_intelligent.txt`
2. **Start Command**: `python run_intelligent_bot.py`
3. **Environment Variables**: 
   ```
   ALPACA_API_KEY=your_key
   ALPACA_SECRET_KEY=your_secret  
   ALPACA_BASE_URL=https://paper-api.alpaca.markets
   ENV=production
   LOG_LEVEL=INFO
   PORT=10000
   ```

## Required Environment Variables

### Minimum Required (3 variables):
```bash
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key  
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

### Full Production Stack (7 variables):
```bash
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets
POSTGRES_PASSWORD=secure_password
POSTGRES_URL=your_postgres_connection_string
GRAFANA_PASSWORD=admin_password
ENV=production
```

## Deployment Options

### Minimal Deployment (Single Container)
Use just `Dockerfile.prod` with 3 Alpaca variables:
```bash
docker build -f Dockerfile.prod -t trading-bot .
docker run -d -p 8080:8080 \
  -e ALPACA_API_KEY=your_key \
  -e ALPACA_SECRET_KEY=your_secret \
  -e ALPACA_BASE_URL=https://paper-api.alpaca.markets \
  trading-bot
```

### Full Production Stack (Multi-Container)  
Use `docker-compose.prod.yml` with full environment:
```bash
docker-compose -f docker-compose.prod.yml --env-file .env.production up -d
```

## Monitoring & Health Checks

### Health Check Endpoint
```bash
curl http://localhost:8080/health
```

### Key Metrics to Monitor
- **CPU Usage**: Should stay below 80%
- **Memory Usage**: Should stay below 1GB
- **Trade Execution Latency**: Should be under 100ms
- **Win Rate**: Should maintain above 45%
- **Daily Drawdown**: Should stay below 5%

### Grafana Dashboards
Access Grafana at http://localhost:3000 with:
- Username: `admin`
- Password: `your_grafana_password`

Pre-configured dashboards include:
- Trading Performance Overview
- Risk Management Metrics
- System Resource Usage
- Alert Management

## Scaling & High Availability

### Horizontal Scaling
```bash
# Scale trading bot instances
docker-compose -f docker-compose.prod.yml up -d --scale trading-bot=3
```

### Database Backup
```bash
# Backup PostgreSQL data
docker exec trading-postgres pg_dump -U trading trading_bot > backup.sql
```

### Log Management
```bash
# View real-time logs
docker-compose -f docker-compose.prod.yml logs -f

# Archive logs
docker-compose -f docker-compose.prod.yml logs --no-color > trading_bot_$(date +%Y%m%d).log
```

## Troubleshooting

### Common Issues

1. **"Permission denied" errors**
   ```bash
   # Fix file permissions
   chmod +x run_intelligent_bot.py
   ```

2. **Database connection failed**
   ```bash
   # Check PostgreSQL is running
   docker-compose -f docker-compose.prod.yml ps postgres
   ```

3. **High memory usage**
   ```bash
   # Monitor resource usage
   docker stats trading-bot
   ```

4. **Trading API errors**
   - Verify API keys in environment variables
   - Check Alpaca account status
   - Ensure sufficient buying power

### Emergency Procedures

#### Stop Trading (Circuit Breaker)
```bash
# Manual circuit breaker activation
curl -X POST http://localhost:8080/api/emergency/stop
```

#### Force Container Restart
```bash
docker-compose -f docker-compose.prod.yml restart trading-bot
```

#### Emergency Shutdown
```bash
docker-compose -f docker-compose.prod.yml down
```

## Security Best Practices

1. **Never commit API keys** to version control
2. **Use strong passwords** for database and Grafana
3. **Enable HTTPS** in production
4. **Regularly update** Docker images
5. **Monitor logs** for suspicious activity
6. **Backup data** regularly

## Performance Tuning

### Memory Optimization
```bash
# Limit container memory
docker run --memory=1g trading-bot
```

### CPU Optimization  
```bash
# Limit CPU usage
docker run --cpus="1.5" trading-bot
```

### Database Optimization
- Use connection pooling
- Regular vacuum and analyze
- Monitor query performance