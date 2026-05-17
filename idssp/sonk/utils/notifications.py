"""idssp/sonk/utils/notifications.py
Lightweight, fire-and-forget HTTP webhook dispatcher for training alerts.
Uses HTTPS (port 443) to bypass cloud SMTP restrictions.
Supports text-only or text+file attachment.
"""
import html
import logging
import threading
from pathlib import Path

import requests

from idssp.sonk import config

logger = logging.getLogger(__name__)


def _escape_html_for_telegram(text: str) -> str:
    """Escape HTML-special chars while preserving allowed Telegram tags."""
    # Temporarily protect allowed tags
    protected = text
    for tag in ["b", "i", "u", "s", "code", "pre", "a"]:
        protected = protected.replace(f"<{tag}>", f"__TAG_OPEN_{tag}__")
        protected = protected.replace(f"</{tag}>", f"__TAG_CLOSE_{tag}__")

    # Escape remaining special chars
    escaped = html.escape(protected, quote=False)

    # Restore allowed tags
    for tag in ["b", "i", "u", "s", "code", "pre", "a"]:
        escaped = escaped.replace(f"__TAG_OPEN_{tag}__", f"<{tag}>")
        escaped = escaped.replace(f"__TAG_CLOSE_{tag}__", f"</{tag}>")

    return escaped

def send_alert(title: str, message: str, sync: bool = False, file_path: str | None = None, timeout: float = 10.0) -> None:
    """
    Dispatches a formatted alert via Telegram.
    
    Args:
        title: Bold header for the notification.
        message: Body text (supports HTML tags like <code>, <b>, etc.).
        sync:  If True, blocks until delivery completes. Use at shutdown.
               If False (default), runs in a daemon thread. Safe mid-training.
        file_path: Optional path to attach a file (e.g., training log).
        timeout: Maximum time to wait for the request to complete.
    """
    if not getattr(config, "ENABLE_TELEGRAM_ALERTS", False):
        return

    bot_token = config.TELEGRAM_BOT_TOKEN
    chat_id = config.TELEGRAM_CHAT_ID
    base_url = f"https://api.telegram.org/bot{bot_token}"

    def _post():
        try:
            safe_title = _escape_html_for_telegram(title)
            safe_message = _escape_html_for_telegram(message)
            if file_path:
                path = Path(file_path)
                if not path.exists():
                    logger.error("File not found for attachment: %s", file_path)
                    return

                # Telegram limit is 50 MB; we'll cap at 45 MB to leave headroom
                if path.stat().st_size > 45 * 1024 * 1024:
                    logger.warning("File exceeds 45 MB. Skipping attachment: %s", file_path)
                    return

                endpoint = f"{base_url}/sendDocument"
                payload = {
                    "chat_id": chat_id,
                    "caption": f"<b>{safe_title}</b>\n{safe_message}",
                    "parse_mode": "HTML"
                }
                with open(path, "rb") as f:
                    # requests handles multipart/form-data automatically
                    resp = requests.post(
                        endpoint,
                        data=payload,
                        files={"document": (path.name, f)},
                        timeout=timeout)
            else:
                endpoint = f"{base_url}/sendMessage"
                payload = {
                    "chat_id": chat_id,
                    "text": f"<b>{safe_title}</b>\n{safe_message}",
                    "parse_mode": "HTML"
                }
                resp = requests.post(endpoint, json=payload, timeout=timeout)

            resp.raise_for_status()
            logger.debug("Telegram alert dispatched successfully.")
        except requests.RequestException as e:
            logger.error("Telegram alert failed: %s", e)

    if sync:
        _post()
    else:
        threading.Thread(target=_post, daemon=True, name="alert-dispatcher").start()
