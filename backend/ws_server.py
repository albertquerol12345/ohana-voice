import asyncio
import json

import websockets


class WSBroadcaster:
    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self.queue: asyncio.Queue = asyncio.Queue()
        self.clients: set = set()

    def publish(self, payload: dict):
        self.loop.call_soon_threadsafe(self.queue.put_nowait, payload)

    async def handler(self, websocket):
        self.clients.add(websocket)
        await websocket.send(json.dumps({"type": "state", "state": "listening"}))
        try:
            async for _ in websocket:
                pass
        finally:
            self.clients.discard(websocket)

    async def broadcaster(self):
        while True:
            payload = await self.queue.get()
            if not self.clients:
                continue
            data = json.dumps(payload, ensure_ascii=False)
            stale = []
            for ws in self.clients:
                try:
                    await ws.send(data)
                except Exception:
                    stale.append(ws)
            for ws in stale:
                self.clients.discard(ws)

    async def serve(self, host: str, port: int):
        async with websockets.serve(self.handler, host, port):
            await self.broadcaster()
