# CodeMind API

FastAPI backend for CodeMind - the world's first cognitive software factory that transforms business ideas into production-ready applications using advanced AI.

## 🚀 Features

### 🧠 Intelligent LLM Routing
- **Smart Model Selection**: Automatically routes requests to optimal models based on task complexity and cost
- **Multi-Provider Support**: AWS Bedrock (Claude 3.7, Claude 3 Haiku, Claude 3 Sonnet) + Azure OpenAI (GPT-5, GPT-4.1 family, O4-mini)
- **Cost Optimization**: Real-time cost tracking and performance monitoring
- **UK Region Optimized**: London AWS (eu-west-2) and UK South Azure

### 🏗️ Enterprise Architecture
- **Production-Ready**: Modular FastAPI structure with proper separation of concerns
- **Enterprise Middleware**: Authentication, rate limiting, context tracking, request IDs
- **Health Monitoring**: `/health` and `/health/detailed` endpoints
- **Error Handling**: Graceful degradation and comprehensive error responses

### 📊 Database Integration
- **PostgreSQL**: Primary database with SQLAlchemy async ORM
- **Redis**: Caching and session management
- **Qdrant**: Vector database for semantic search
- **NATS**: Messaging and event streaming

## 🛠️ Quick Start

### Prerequisites
- Python 3.10+
- Kubernetes cluster with CodeMind infrastructure deployed
- AWS Bedrock access (eu-west-2 region)
- Azure OpenAI access (optional)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/QuantumLayerPlatform-dev-ai/codemind-api.git
   cd codemind-api
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials and K8s node IP
   ```

5. **Run the server**
   ```bash
   python app.py
   ```

The API will be available at `http://localhost:8000`

## 📚 API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# UK Region Settings
AWS_REGION=eu-west-2
AZURE_REGION=uksouth

# Database (NodePort K8s services)
DATABASE_URL=postgresql+asyncpg://postgres:password@k8s-node-ip:30432/codemind
REDIS_URL=redis://:password@k8s-node-ip:30379
QDRANT_URL=http://k8s-node-ip:30333

# AWS Bedrock (London region)
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret

# Azure OpenAI (UK South)
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key
```

### Available Models

**AWS Bedrock (eu-west-2)**:
- `anthropic.claude-3-7-sonnet-20250219-v1:0` - Latest Claude 3.7 Sonnet
- `anthropic.claude-3-haiku-20240307-v1:0` - Fast Claude 3 Haiku
- `anthropic.claude-3-sonnet-20240229-v1:0` - Claude 3 Sonnet

**Azure OpenAI (uksouth)**:
- GPT-5, GPT-4.1, GPT-4.1-mini, GPT-4.1-nano, O4-mini (based on your deployments)

## 🎯 API Endpoints

### Business Intent Analysis
```bash
POST /api/v1/generate/app
```
Analyzes business descriptions and generates comprehensive business plans.

### Code Generation
```bash
POST /api/v1/generate/code
```
Generates production-ready code based on business requirements.

### Health Monitoring
```bash
GET /health
GET /health/detailed
```

## 🔒 Security Features

- **JWT Authentication**: Secure API access
- **Rate Limiting**: Prevents abuse and ensures fair usage
- **Request Tracking**: Unique request IDs for monitoring
- **Input Validation**: Comprehensive request validation
- **CORS Configuration**: Secure cross-origin requests

## 📈 Performance Monitoring

- **Real-time Metrics**: Response times, token usage, costs
- **Model Performance**: Confidence scores and success rates
- **Error Tracking**: Comprehensive error logging and monitoring
- **Cost Optimization**: Smart routing based on task complexity

## 🏗️ Architecture

```
CodeMind API
├── api/v1/endpoints/     # API route handlers
├── core/                 # Core configurations
├── middleware/           # Authentication, rate limiting, context
├── models/              # Pydantic models and database schemas
├── services/            # Business logic (LLM router, etc.)
└── app.py              # FastAPI application entry point
```

## 🚀 Production Deployment

The API is designed for Kubernetes deployment with:
- **NodePort Services**: Stable access without port-forwarding
- **Health Checks**: Kubernetes-ready health endpoints
- **Graceful Shutdown**: Proper cleanup of connections
- **Environment-based Configuration**: Easy container deployment

## 📊 Current Status

- ✅ **Production Ready**: Complete enterprise-grade implementation
- ✅ **Live System**: Running successfully on localhost:8000
- ✅ **Real LLM Integration**: AWS Bedrock Claude 3.7 working
- ✅ **UK Compliance**: London and UK South regions
- ✅ **Cost Effective**: ~$0.006 per business analysis

## 🤝 Contributing

This is part of the CodeMind cognitive software factory. See the main repository for contribution guidelines.

## 📄 License

Commercial license - see LICENSE for details.

---

**🇬🇧 Built in the UK, for the UK, ready for the world! 🚀**

Part of the [CodeMind](https://github.com/QuantumLayerPlatform-dev-ai/codemind) ecosystem.