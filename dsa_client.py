"""
dsa_client.py
-------------
A Python client for the AI-Interview-Bot (Node.js DSA bot) running on port 3000.
Call run_interactive_session() to start a full DSA interview in the terminal.

Usage (standalone):
    python dsa_client.py

Usage (from arch.py):
    from dsa_client import DSABotClient
    import asyncio
    client = DSABotClient()
    asyncio.run(client.run_interactive_session("some-user-id"))
"""

import httpx
import asyncio
import uuid

DSA_BOT_BASE_URL = "http://localhost:3000"


class DSABotClient:
    def __init__(self, base_url: str = DSA_BOT_BASE_URL):
        self.base_url = base_url

    async def _get(self, path: str) -> dict:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{self.base_url}{path}")
            resp.raise_for_status()
            return resp.json()

    async def _post(self, path: str, body: dict) -> dict:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{self.base_url}{path}", json=body)
            resp.raise_for_status()
            return resp.json()

    async def start_session(self, user_id: str) -> dict:
        """Initialize a fresh DSA interview session. Returns {userId, stage, reply}."""
        return await self._get(f"/start/{user_id}")

    async def get_status(self, user_id: str) -> dict:
        """Get the current stage of a session. Returns {userId, stage}."""
        return await self._get(f"/status/{user_id}")

    async def send_message(self, user_id: str, message: str = "") -> dict:
        """Send a message to the bot. Returns {userId, stage, reply}."""
        return await self._post("/chat", {"userId": user_id, "message": message})

    async def run_interactive_session(self, user_id: str | None = None):
        """
        Run a full interactive DSA interview session in the terminal.
        Handles the full conversational loop until the session is COMPLETE.
        """
        if user_id is None:
            user_id = str(uuid.uuid4())

        print("\n" + "=" * 60)
        print("🧩  DSA INTERVIEW BOT  (powered by AI-Interview-Bot)")
        print("=" * 60)

        # Check the server is reachable
        try:
            data = await self.start_session(user_id)
        except httpx.ConnectError:
            print(
                "\n❌  Could not connect to the DSA bot server at "
                f"{self.base_url}\n"
                "   Make sure it is running:\n"
                "   cd /Users/umanggoel/Desktop/Projects/DSA_Agent\n"
                "   node server.js\n"
            )
            return

        print(f"\n{data.get('reply', '')}\n")
        current_stage = data.get("stage")

        # The first /start already moves us into SHOW_QUESTION.
        # We now need to POST /chat to get the actual problem shown.
        data = await self.send_message(user_id, "")
        print(f"\n{data.get('reply', '')}\n")
        current_stage = data.get("stage")

        while current_stage != "COMPLETE":
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n👋  Exiting DSA interview.")
                break

            data = await self.send_message(user_id, user_input)
            reply = data.get("reply", "")
            current_stage = data.get("stage")

            print(f"\nBot: {reply}\n")

            if current_stage == "COMPLETE":
                print("✅  DSA Interview complete!")
                break


# ── Standalone entry-point ──────────────────────────────────────────────────

if __name__ == "__main__":
    client = DSABotClient()
    asyncio.run(client.run_interactive_session())
