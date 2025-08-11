# main.py
from fastapi import FastAPI
from routes.routes import router as all_routes

app = FastAPI()
app.include_router(all_routes)