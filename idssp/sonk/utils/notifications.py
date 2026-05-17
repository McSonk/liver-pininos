"""idssp/sonk/utils/notifications.py
Lightweight, fire-and-forget HTTP webhook dispatcher for training alerts.
Uses HTTPS (port 443) to bypass cloud SMTP restrictions.
"""
import logging
import threading

import requests

from idssp.sonk import config

logger = logging.getLogger(__name__)

def send_alert(title: str, message: str, sync: bool = False, timeout: float = 10.0) -> None:
    """
    Dispatches a formatted alert via Telegram.
    
    Args:
        title: Bold header for the notification.
        message: Body text (supports HTML tags like <code>, <b>, etc.).
        sync:  If True, blocks until delivery completes (use at shutdown).
               If False (default), runs in a daemon thread (safe mid-training).
        timeout: Maximum time (in seconds) to wait for the request to complete.
    """
    if not getattr(config, "ENABLE_TELEGRAM_ALERTS", False):
        return

    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": config.TELEGRAM_CHAT_ID,
        "text": f"<b>{title}</b>\n{message}",
        "parse_mode": "HTML"
    }

    def _post():
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            logger.debug("Telegram alert dispatched successfully.")
        except requests.RequestException as e:
            logger.error("Telegram alert failed: %s", e)

    if sync:
        _post()  # Run in main thread (guaranteed delivery)
    else:
        threading.Thread(target=_post, daemon=True, name="alert-dispatcher").start()
