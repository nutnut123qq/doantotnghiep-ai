from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import summarize, analyze, forecast, qa, alert_nlp, stock_data

app = FastAPI(title="Stock Investment AI Service", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(summarize.router, prefix="/api", tags=["summarize"])
app.include_router(analyze.router, prefix="/api", tags=["analyze"])
app.include_router(forecast.router, prefix="/api", tags=["forecast"])
app.include_router(qa.router, prefix="/api", tags=["qa"])
app.include_router(alert_nlp.router, prefix="/api", tags=["alert-nlp"])
app.include_router(stock_data.router, tags=["stock"])


@app.get("/")
async def root():
    return {"message": "Stock Investment AI Service"}


@app.get("/health")
async def health():
    return {"status": "healthy"}

