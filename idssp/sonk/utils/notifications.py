"""idssp/sonk/utils/notifications.py
Lightweight, fire-and-forget HTTP webhook dispatcher for training alerts.
Uses HTTPS (port 443) to bypass cloud SMTP restrictions.
Supports text-only or text+file attachment with safe truncation.
"""
import html
import re
import threading
from pathlib import Path

import requests

from idssp.sonk import config
from idssp.sonk.utils.logger import get_logger

logger = get_logger(__name__)

# Telegram API limits (per https://core.telegram.org/bots/api)
_TELEGRAM_TEXT_LIMIT = 4096      # for sendMessage text field
_TELEGRAM_CAPTION_LIMIT = 1024   # for sendDocument caption field
_TRUNCATION_INDICATOR = "… (truncated)"
# Regex to match Telegram bot tokens, including tokens embedded in Telegram API URLs
_BOT_TOKEN_PATTERN = re.compile(r"(?<!\d)\d+:[A-Za-z0-9_-]+\b")

def _redact_bot_token(text: str, placeholder: str = "[---]") -> str:
    """Redacts Telegram bot tokens from any string."""
    return _BOT_TOKEN_PATTERN.sub(placeholder, text)

def _escape_html_for_telegram(text: str) -> str:
    """Escape HTML-special chars while preserving allowed Telegram tags."""
    # Temporarily protect only exact tags that are valid in this literal form.
    # Do not preserve bare <a></a>: Telegram anchors require an href attribute.
    protected = text
    for tag in ["b", "i", "u", "s", "code", "pre"]:
        protected = protected.replace(f"<{tag}>", f"__TAG_OPEN_{tag}__")
        protected = protected.replace(f"</{tag}>", f"__TAG_CLOSE_{tag}__")

    # Escape remaining special chars
    escaped = html.escape(protected, quote=False)

    # Restore allowed exact tags
    for tag in ["b", "i", "u", "s", "code", "pre"]:
        escaped = escaped.replace(f"__TAG_OPEN_{tag}__", f"<{tag}>")
        escaped = escaped.replace(f"__TAG_CLOSE_{tag}__", f"</{tag}>")

    return escaped


def _truncate_for_telegram(title: str, message: str, is_caption: bool = False) -> tuple[str, str]:
    """
    Truncates title + message to fit Telegram's character limits.
    Guarantees: len(f"<b>{safe_title}</b>\n{safe_message}") <= limit
    
    Args:
        title: The notification title (truncated if needed).
        message: The message body (truncated if needed).
        is_caption: If True, use 1024-char caption limit; else use 4096-char text limit.
    
    Returns:
        (safe_title, safe_message) tuple, both HTML-escaped and within limits.
    """
    limit = _TELEGRAM_CAPTION_LIMIT if is_caption else _TELEGRAM_TEXT_LIMIT

    # Escape first so we count the actual characters that will be sent
    safe_title = _escape_html_for_telegram(title)
    safe_message = _escape_html_for_telegram(message)

    # Calculate the formatted title overhead: <b>title</b>\n
    title_wrapper = f"<b>{safe_title}</b>\n"

    # === STEP 1: Handle overlong title ===
    if len(title_wrapper) >= limit:
        # Title alone exceeds limit; truncate title to fit with minimal message space
        # Reserve space for </b>\n + truncation indicator + at least 1 char of message
        min_message_space = len("</b>\n") + len(_TRUNCATION_INDICATOR) + 1
        max_title_len = limit - min_message_space - len("<b>")  # Account for opening tag

        if max_title_len <= 0:
            # Pathological case: limit is too small for any content
            # Return minimal viable payload
            return _escape_html_for_telegram("Alert"), _TRUNCATION_INDICATOR

        truncated_title = safe_title[:max_title_len]
        # Avoid breaking mid-entity or mid-tag
        if truncated_title.endswith('&') or truncated_title.endswith('&l') or truncated_title.endswith('&lt'):
            truncated_title = truncated_title.rsplit('&', 1)[0]
        if truncated_title.endswith('<') or truncated_title.endswith('</'):
            truncated_title = truncated_title.rsplit('<', 1)[0]
        truncated_title += _TRUNCATION_INDICATOR

        logger.warning(
            "Telegram title truncated to %d chars (limit: %d). Original title length: %d",
            len(f"<b>{truncated_title}</b>\n"), limit, len(title_wrapper)
        )
        # Message gets only the truncation indicator since title consumed most space
        return truncated_title, _TRUNCATION_INDICATOR

    # === STEP 2: Title fits; allocate remaining space to message ===
    overhead = len(title_wrapper)
    available_for_message = limit - overhead - len(_TRUNCATION_INDICATOR)

    if available_for_message <= 0:
        # Edge case: title + formatting leaves no room for message
        logger.warning(
            "Telegram message omitted: title consumes %d/%d chars",
            overhead, limit
        )
        return safe_title, _TRUNCATION_INDICATOR

    if len(safe_message) <= available_for_message:
        return safe_title, safe_message

    # Truncate message safely
    truncated_msg = safe_message[:available_for_message]

    # Avoid cutting mid-entity (e.g., &lt;) or mid-tag
    if truncated_msg.endswith('&') or truncated_msg.endswith('&l') or truncated_msg.endswith('&lt'):
        truncated_msg = truncated_msg.rsplit('&', 1)[0]
    if truncated_msg.endswith('<') or truncated_msg.endswith('</'):
        truncated_msg = truncated_msg.rsplit('<', 1)[0]

    truncated_msg += _TRUNCATION_INDICATOR
    logger.warning(
        "Telegram message truncated to %d chars (limit: %d). Original message length: %d",
        len(f"{title_wrapper}{truncated_msg}"), limit, len(f"{title_wrapper}{safe_message}")
    )

    return safe_title, truncated_msg


def send_alert(title: str, message: str, sync: bool = False, file_path: str = None, timeout: float = 10.0) -> None:
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
    if not config.ENABLE_TELEGRAM_NOTIFICATIONS:
        logger.debug("Telegram notifications are disabled. Skipping alert.")
        return

    bot_token = config.TELEGRAM_BOT_TOKEN
    chat_id = config.TELEGRAM_CHAT_ID
    base_url = f"https://api.telegram.org/bot{bot_token}"

    def _send_file(title, message):
        path = Path(file_path)
        if not path.exists():
            logger.error("File not found for attachment: %s", file_path)
            raise FileNotFoundError(f"Attachment file not found: {file_path}")

        # Telegram limit is 50 MB; we'll cap at 45 MB to leave headroom
        if path.stat().st_size > 45 * 1024 * 1024:
            logger.warning("File exceeds 45 MB. Skipping attachment: %s", file_path)
            raise ValueError("Attachment file too large for Telegram (max 45 MB)")

        endpoint = f"{base_url}/sendDocument"
        payload = {
            "chat_id": chat_id,
            "caption": f"<b>{title}</b>\n{message}",
            "parse_mode": "HTML"
        }
        with open(path, "rb") as f:
            # requests handles multipart/form-data automatically
            resp = requests.post(
                endpoint,
                data=payload,
                files={"document": (path.name, f)},
                timeout=timeout)
        return resp

    def _send_message(title, message):
        endpoint = f"{base_url}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": f"<b>{title}</b>\n{message}",
            "parse_mode": "HTML"
        }
        return requests.post(endpoint, json=payload, timeout=timeout)

    def _post():
        try:
            # Apply truncation BEFORE sending
            if file_path:
                safe_title, safe_message = _truncate_for_telegram(title, message, is_caption=True)
            else:
                safe_title, safe_message = _truncate_for_telegram(title, message, is_caption=False)

            if file_path:
                try:
                    resp = _send_file(safe_title, safe_message)
                except (FileNotFoundError, ValueError) as e:
                    logger.error("Error occurred while sending file: %s", e)
                    # Fallback to text-only with appropriate limit
                    safe_title, safe_message = _truncate_for_telegram(title, message, is_caption=False)
                    resp = _send_message(safe_title, safe_message)
            else:
                resp = _send_message(safe_title, safe_message)

            resp.raise_for_status()
            logger.info("Telegram alert dispatched successfully.")
        except requests.RequestException as e:
            # Redact sensitive info before logging
            error_msg = _redact_bot_token(str(e))

            # Try to extract Telegram's structured error response
            error_details = "No response body"
            if e.response is not None:
                try:
                    resp_json = e.response.json()
                    # Telegram returns {"ok": false, "error_code": 400, "description": "..."}
                    error_details = resp_json.get("description", str(resp_json))
                except Exception:
                    # Fallback if response isn't valid JSON
                    error_details = _redact_bot_token(e.response.text[:200])

            # Log a safe message
            logger.error(
                "Telegram alert failed [status: %s]: %s | API response: %s",
                e.response.status_code if e.response else "N/A",
                error_msg,
                error_details
            )

        except Exception as e:
            logger.error("Telegram alert failed due to an unexpected error: %s", e)

    if sync:
        _post()
    else:
        threading.Thread(target=_post, daemon=True, name="alert-dispatcher").start()
