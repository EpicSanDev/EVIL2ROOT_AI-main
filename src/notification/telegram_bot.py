import os
from telegram import Bot
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TelegramBot:
    def __init__(self):
        token = os.getenv('TELEGRAM_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.enabled = False
        
        if token and self.chat_id:
            try:
                self.bot = Bot(token=token)
                self.enabled = True
            except Exception as e:
                print(f"Failed to initialize Telegram bot (notifications disabled): {e}")
                self.bot = None
        else:
            print("Telegram bot not configured (notifications disabled)")
            self.bot = None

    async def send_message(self, message):
        if not self.enabled:
            print(f"[Notification]: {message}")
            return
            
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode='Markdown')
        except Exception as e:
            print(f"[Notification]: {message}")
            print(f"Error sending Telegram message: {e}")
            self.enabled = False  # Disable on error
