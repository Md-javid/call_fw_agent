import os
import json
import asyncio
import requests
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from google import genai
from google.genai import types

app = FastAPI()

# Configuration from Environment Variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

client = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM_INSTRUCTION = (
    "You are a helpful, brief personal assistant handling a missed phone call. "
    "The user cannot answer right now. Natively match the language, dialect, and accent "
    "of the caller instantly. If they speak Hindi, reply in Hindi. If Tamil, reply in Tamil. "
    "Ask for their name and message, keep responses under 2 sentences, and say goodbye."
)

def send_telegram_summary(caller_number, summary_text):
    """Sends the formatted English summary to your Telegram app for free"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram environment variables are missing. Skipping notification.")
        return

    url = f"https://telegram.org{TELEGRAM_BOT_TOKEN}/sendMessage"
    message_content = (
        f"📞 *Missed Call Personal Assistant*\n"
        f"👤 *From:* {caller_number}\n\n"
        f"📝 *Summary (Translated to English):*\n{summary_text}"
    )
    
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message_content,
        "parse_mode": "Markdown"
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            print(f"Failed to send Telegram message: {response.text}")
    except Exception as e:
        print(f"Error firing Telegram webhook: {e}")

@app.api_route("/twiml", methods=["GET", "POST"])
async def route_twiml():
    host = os.getenv("RENDER_EXTERNAL_HOSTNAME", "localhost:8000")
    protocol = "wss" if "render" in host else "ws"
    
    twiml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Connect>
            <Stream url="{protocol}://{host}/media-stream" />
        </Connect>
    </Response>
    """
    return HTMLResponse(content=twiml_response, media_type="application/xml")

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    await websocket.accept()
    stream_sid = None
    caller_number = "Unknown Caller"
    
    # In-memory accumulator to store the conversation history text chunks
    conversation_transcript = []
    
    async with client.aio.live.connect(
        model="gemini-2.0-flash-exp",
        config=types.LiveConnectConfig(
            response_modalities=[types.LiveModality.AUDIO],
            system_instruction=types.Content(parts=[types.Part.from_text(SYSTEM_INSTRUCTION)]),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Aoede")
                )
            )
        )
    ) as session:

        async def send_to_gemini():
            nonlocal stream_sid, caller_number
            try:
                async for message in websocket.iter_text():
                    data = json.loads(message)
                    if data['event'] == 'start':
                        stream_sid = data['start']['streamSid']
                        # Capture caller details from Twilio's metadata if available
                        if 'customParameters' in data['start']:
                            caller_number = data['start'].get('from', caller_number)
                    elif data['event'] == 'media':
                        audio_payload = data['media']['payload']
                        await session.send(
                            input={"data": audio_payload, "mime_type": "audio/pcm;rate=8000"},
                            end_of_turn=False
                        )
            except Exception as e:
                print(f"Inbound stream metadata tracking closed: {e}")

        async def receive_from_gemini():
            try:
                async for response in session.receive():
                    server_content = response.server_content
                    if server_content is not None and server_content.model_turn is not None:
                        for part in server_content.model_turn.parts:
                            # 1. Handle live voice playback to the caller
                            if part.inline_data is not None:
                                twilio_message = {
                                    "event": "media",
                                    "streamSid": stream_sid,
                                    "media": {"payload": part.inline_data.data}
                                }
                                await websocket.send_text(json.dumps(twilio_message))
                            
                            # 2. Extract textual interpretation to build the backup transcript log
                            if part.text is not None:
                                conversation_transcript.append(f"AI: {part.text}")
            except Exception as e:
                print(f"Outbound streaming tracking channel terminated: {e}")

        # Block here while the live WebRTC loop is operational
        await asyncio.gather(send_to_gemini(), receive_from_gemini())
        
        # --- POST-CALL TRIGGER WORKFLOW ---
        # Executed instantly when the websocket disconnects (caller hangs up)
        if conversation_transcript:
            full_raw_log = "\n".join(conversation_transcript)
            
            # Request Gemini to summarize the session into English text
            summary_prompt = (
                f"Review the following dialogue history from a phone call application. "
                f"The caller may have spoken in an Indian regional language (like Tamil or Hindi) or English. "
                f"Translate their statements completely into English if necessary, and write a highly concise, "
                f"bulleted summary explaining who they are, the core reason they called, and any clear actionable tasks.\n\n"
                f"Log:\n{full_raw_log}"
            )
            
            try:
                summary_response = client.models.generate_content(
                    model="gemini-2.0-flash-exp",
                    contents=summary_prompt,
                )
                # Ship the final processed output text to Telegram
                send_telegram_summary(caller_number, summary_response.text)
            except Exception as summary_error:
                print(f"Failed to generate English transcript summary: {summary_error}")
