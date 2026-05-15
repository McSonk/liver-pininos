# mail.py
import smtplib
import socket
import threading
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from idssp.sonk import config
from idssp.sonk.utils.logger import get_logger

logger = get_logger(__name__)


def send_training_email(
    subject: str,
    body: str,
    log_file: Path = None,
    wait_for_completion: bool = False,
    timeout: float = 20.0
) -> None:
    """
    Sends an email notification with the log file attached.
    
    Params
    ------
    `subject`: str
        Email subject line.
    `body`: str
        Plain-text email body.
    `log_file`: Path, optional
        Path to the log file to attach. If None or missing, email is sent without attachment.
    `wait_for_completion`: bool, default False
        If True, blocks the caller for up to `timeout` seconds to ensure the email is sent.
        Use True for critical notifications (interrupt/failure), False for non-critical (start/success).
    `timeout`: float, default 20.0
        Maximum seconds to wait for the email thread to complete (only used if wait_for_completion=True).
    """
    def _send_async():
        if not config.ENABLE_EMAIL_NOTIFICATIONS:
            return

        creds = [config.EMAIL_SENDER, config.EMAIL_PASSWORD, config.EMAIL_RECIPIENT]
        if not all(creds):
            logger.warning("Missing credentials. Skipping notification.")
            return

        msg = MIMEMultipart()
        msg['From'] = config.EMAIL_SENDER
        msg['To'] = config.EMAIL_RECIPIENT
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        try:
            if log_file is not None and log_file.exists():
                with open(log_file, 'rb') as f:
                    attachment = MIMEApplication(f.read(), Name=log_file.name)
                attachment['Content-Disposition'] = f'attachment; filename="{log_file.name}"'
                msg.attach(attachment)
            elif log_file is not None:
                logger.warning("Log file not found: %s", log_file)

            with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=15) as server:
                server.starttls()
                server.login(config.EMAIL_SENDER, config.EMAIL_PASSWORD)
                server.send_message(msg)

            logger.info("Notification sent: %s", subject)

        except FileNotFoundError:
            logger.error("Log file missing: %s", log_file)
        except smtplib.SMTPAuthenticationError:
            logger.error("Auth failed. Use a Gmail App Password, not account password.")
        except socket.timeout:
            logger.error("SMTP connection timed out (15s). Check network/SMTP_HOST.")
        except smtplib.SMTPException as e:
            logger.error("SMTP protocol error: %s", e)
        except Exception as e:
            logger.error("Unexpected error during email dispatch: %s", e, exc_info=True)

    # Use daemon=False so the thread can complete during shutdown if we wait for it
    thread = threading.Thread(target=_send_async, daemon=not wait_for_completion)
    thread.start()

    if wait_for_completion:
        # Block briefly to allow the email to send before process exit
        thread.join(timeout=timeout)
        if thread.is_alive():
            logger.warning("Email thread did not complete within %ds timeout", timeout)