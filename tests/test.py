import asyncio
import wave
import websockets


async def test():
    with wave.open("test.wav", "rb") as f:
        pcm = f.readframes(f.getnframes())

    async with websockets.connect("ws://localhost:8080/ws") as ws:
        chunk_size = 512 * 2  # 512 samples × 2 bytes i16 LE
        chunks = [pcm[i : i + chunk_size] for i in range(0, len(pcm), chunk_size)]
        print(f"sending {len(chunks)} chunks at real-time pace")
        for chunk in chunks:
            if len(chunk) == chunk_size:
                await ws.send(chunk)
                await asyncio.sleep(0.032)  # 512 samples @ 16kHz = 32ms
        print("done")


asyncio.run(test())
