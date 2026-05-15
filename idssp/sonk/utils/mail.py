import smtplib
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from idssp.sonk import config
from idssp.sonk.utils.logger import get_logger

logger = get_logger(__name__)


def send_training_email(subject: str, body: str, log_file: Path = None) -> bool:
    """
    Sends an email notification with the latest log file attached.
    Gracefully handles missing credentials or SMTP failures.
    """
    if not config.ENABLE_EMAIL_NOTIFICATIONS:
        return False

    msg = MIMEMultipart()
    msg['From'] = config.EMAIL_SENDER
    msg['To'] = config.EMAIL_RECIPIENT
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        # Attach the specified log file
        if log_file is not None:
            with open(log_file, 'rb') as f:
                attachment = MIMEApplication(f.read(), Name=log_file.name)
            attachment['Content-Disposition'] = f'attachment; filename="{log_file.name}"'
            msg.attach(attachment)

        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT) as server:
            server.starttls()
            server.login(config.EMAIL_SENDER, config.EMAIL_PASSWORD)
            server.send_message(msg)
        logger.info("Notification sent: %s", subject)
        return True
    except FileNotFoundError:
        logger.error("Log file missing: %s", log_file)
        return False
    except smtplib.SMTPAuthenticationError:
        logger.error("SMTP auth failed. Check EMAIL_PASSWORD (use App Password for Gmail).")
        return False
    except Exception as e:
        logger.error("Failed to send: %s", e, exc_info=True)
        return False
