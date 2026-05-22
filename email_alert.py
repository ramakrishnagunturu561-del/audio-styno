import smtplib
from email.message import EmailMessage
import os

def send_stego_email(sender_email, app_password, receiver_email, file_path):
    """
    Sends an email with the generated steganography audio file attached.
    Uses SMTP starttls on port 587. Default configured for Gmail.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Attachment file not found: {file_path}")

    msg = EmailMessage()
    msg['Subject'] = "Secure Audio Transfer"
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg.set_content("Attached is the secured audio steganography file. Please decode it using the agreed password.")

    # Attach the .wav file
    with open(file_path, 'rb') as f:
        file_data = f.read()
        file_name = os.path.basename(file_path)

    # Maintype audio, subtype wav
    msg.add_attachment(file_data, maintype='audio', subtype='wav', filename=file_name)

    # Connect to SMTP server and send
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.ehlo()
        server.starttls() # Secure connection
        server.ehlo()
        server.login(sender_email, app_password)
        server.send_message(msg)
    except Exception as e:
        raise Exception(f"Failed to send email: {e}")
    finally:
        try:
            server.quit()
        except:
            pass
