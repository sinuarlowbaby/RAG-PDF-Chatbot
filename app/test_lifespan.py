import asyncio
from app import lifespan
from fastapi import FastAPI
import logging

logging.basicConfig(level=logging.INFO)
app = FastAPI()

async def main():
    try:
        async with lifespan(app) as _:
            print("Lifespan started successfully")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
