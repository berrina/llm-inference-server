from fastapi import FastAPI
from pydantic import BaseModel
from server.batching import scheduler
from server.continuous_batching import continuous_scheduler
from server.metrics import get_stats
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from server.model_runner import model, tokenizer
import torch 
import asyncio

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(scheduler.run_forever())
    asyncio.create_task(continuous_scheduler.run_forever())

@app.post("/generate")
async def generate_endpoint(req: GenerateRequest):
    result = await scheduler.submit(req.prompt, req.max_new_tokens)
    return {"completion": result}

@app.post("/generate_continuous")
async def generate_continuous_endpoint(req: GenerateRequest):
    result = await continuous_scheduler.submit(req.prompt, req.max_new_tokens)
    return {"completion": result}

@app.get("/metrics")
def metrics_endpoint():
    return {
        "static_batching": get_stats("static_batching"),
        "continuous_batching": get_stats("continuous_batching"),
    }

@app.get("/status")
def status_endpoint():
    return {
        "static_queue_depth": scheduler.queue.qsize(),
        "continuous_queue_depth": continuous_scheduler.queue.qsize(),
        "continuous_active_slots": len(continuous_scheduler.active_slots),
    }

@app.post("/generate_stream")
async def generate_stream_endpoint(req: GenerateRequest):
    async def token_generator():
        inputs = tokenizer(req.prompt, return_tensors="pt")
        next_input = inputs.input_ids
        past = None
        for _ in range(req.max_new_tokens):
            with torch.no_grad():
                out = model(input_ids=next_input, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_id = out.logits[:, -1, :].argmax(dim=-1)
            tok_id = next_id.item()
            token_text = tokenizer.decode([tok_id])
            yield f"data: {token_text}\n\n"
            next_input = next_id.unsqueeze(0)
            if tok_id == tokenizer.eos_token_id:
                break
            await asyncio.sleep(0)
    return StreamingResponse(token_generator(), media_type="text/event-stream")