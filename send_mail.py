import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.application import MIMEApplication
from email import encoders
from datetime import date, timedelta
from utils import create_connection, select_yesterday_records, update_processed_records


def send_mail(date_str):
    mail_content = '''
    Отчет по машинам.
    '''
    sender_address = 'carcountingkz@gmail.com'
    sender_pass = 'moika123'
    receiver_address = 'yerbolat101@gmail.com'

    message = MIMEMultipart()
    message['From'] = sender_address
    message['To'] = receiver_address
    message['Subject'] = f'Отчет по машинам. {date_str}'

    message.attach(MIMEText(mail_content, 'plain'))

    attach_file = MIMEApplication(open('cars_report.csv', "rb").read())
    attach_file.add_header('Content-Disposition', 'attachment; filename="cars_report.csv"')
    message.attach(attach_file)

    session = smtplib.SMTP('smtp.gmail.com', 587)
    session.starttls()
    session.login(sender_address, sender_pass)
    text = message.as_string()
    session.sendmail(sender_address, receiver_address, text)
    session.quit()
    print('Mail Sent')


def main():
    conn = create_connection()

    rows = []
    with conn:
        rows = select_yesterday_records(conn)
    if os.path.exists("cars_report.csv"):
        os.remove("cars_report.csv")
    if rows:
        processed_ids = []
        with open("cars_report.csv", "w") as file1:
            file1.write("date_time, file_location\n")
            for row in rows:
                rowid, date_time, file_location = row
                processed_ids.append(str(rowid))
                file1.write(f"{date_time}, {file_location}\n")
        send_mail(str(date.today()-timedelta(days=1)))
        conn = create_connection()
        with conn:
            update_processed_records(conn, ', '.join(processed_ids))


if __name__ == '__main__':
    main()