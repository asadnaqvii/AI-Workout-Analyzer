"""
LiveKit voice-coach agent for fitness-trainer-pose-estimation.
Fetches live rep/sets metrics via HTTP and injects as a system message.
"""
import os
import json
import asyncio
import aiohttp
from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, ChatContext, AutoSubscribe
from livekit.agents.cli import run_app
from livekit.agents.worker import WorkerOptions
from livekit.agents.job import get_current_job_context
from livekit.plugins import silero, deepgram, openai, cartesia

load_dotenv()

METRICS_URL = os.getenv("METRICS_URL", "http://127.0.0.1:5000/latest_metrics")

# Your static system prompt (without metrics)
BASE_SYSTEM = (
    "You are a cheerful fitness coach.\n"
    "You will be given JSON metrics about reps and sets.\n"
    "Rules:\n"
    "• Motivate every 3–5 reps, referencing the current count.\n"
    "• Congratulate on set/workout completion.\n"
    "• Use ≤15 words, like a human trainer."
)

class CoachAgent(Agent):
    def __init__(self, chat_ctx: ChatContext):
        super().__init__(
            instructions=BASE_SYSTEM,
            chat_ctx=chat_ctx,
            vad=silero.VAD.load(),
            stt=deepgram.STT(),
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=cartesia.TTS(model="sonic-english"),
        )

    async def stt_node(self, audio, model_settings):
        job_ctx = get_current_job_context()
        async for event in Agent.default.stt_node(self, audio, model_settings):
            if getattr(event, "text", None):
                try:
                    await job_ctx.room.local_participant.publish_data(
                        json.dumps({"transcript": event.text})
                    )
                except:
                    pass
            yield event

    async def llm_node(self, chat_ctx, tools, model_settings):
        # 1) Fetch latest metrics via HTTP
        latest = None
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(METRICS_URL) as resp:
                    if resp.status == 200:
                        latest = await resp.json()
                        print("✅ Fetched metrics:", latest)
                    else:
                        print(f"⚠️ Metrics HTTP {resp.status}")
        except Exception as e:
            print("⚠️ Error fetching metrics:", e)

        # 2) If we got metrics, insert a system message with them
        if latest:
            metrics_msg = json.dumps(latest)
            chat_ctx.add_message(
                role="system",
                content=f"METRICS: {metrics_msg}"
            )
            # also publish for frontend debug
            job_ctx = get_current_job_context()
            try:
                await job_ctx.room.local_participant.publish_data(
                    json.dumps({"debug_metrics": latest})
                )
            except:
                pass

        # 3) Stream through the LLM
        async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
            # forward voice chunks too
            try:
                job_ctx = get_current_job_context()
                await job_ctx.room.local_participant.publish_data(
                    json.dumps({"coach_chunk": chunk})
                )
            except:
                pass
            yield chunk

async def entrypoint(ctx):
    # Only subscribe to audio
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Build initial context with base system instructions
    initial_ctx = ChatContext()
    initial_ctx.add_message(role="system", content=BASE_SYSTEM)

    coach = CoachAgent(chat_ctx=initial_ctx)
    session = AgentSession(
        vad=coach._vad,
        stt=coach._stt,
        llm=coach._llm,
        tts=coach._tts,
    )
    await session.start(agent=coach, room=ctx.room)

if __name__ == "__main__":
    run_app(WorkerOptions(entrypoint_fnc=entrypoint))
