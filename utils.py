import cv2
import logging
import logging.handlers
import math
import sys
import numpy as np
from datetime import datetime
import sqlite3
from sqlite3 import Error

CONN_NAME = "./main.db"


def save_frame(frame, file_name, flip=True):
    # flip BGR to RGB
    if flip:
        cv2.imwrite(file_name, np.flip(frame, 2))
    else:
        cv2.imwrite(file_name, frame)


def init_logging(to_file=False):
    main_logger = logging.getLogger()

    formatter = logging.Formatter(
        fmt='%(asctime)s.%(msecs)03d %(levelname)-8s [%(name)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    handler_stream = logging.StreamHandler(sys.stdout)
    handler_stream.setFormatter(formatter)
    main_logger.addHandler(handler_stream)

    if to_file:
        handler_file = logging.handlers.RotatingFileHandler("debug.log", maxBytes=1024 * 1024 * 400  # 400MB
                                                            , backupCount=10)
        handler_file.setFormatter(formatter)
        main_logger.addHandler(handler_file)

    main_logger.setLevel(logging.DEBUG)

    return main_logger

#=============================================================================


def distance(x, y, type='euclidian', x_weight=1.0, y_weight=1.0):
    if type == 'euclidian':
        return math.sqrt(float((x[0] - y[0])**2) / x_weight + float((x[1] - y[1])**2) / y_weight)


def get_centroid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)

    cx = x + x1
    cy = y + y1

    return (cx, cy)


def skeleton(img):
    ret, img = cv2.threshold(img, 127, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    while(not done):
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True

    return skel


def create_connection(db_file=CONN_NAME):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Error as e:
        print(e)
    return conn


def insert_into_db(conn, photo_location):
    query = """
            INSERT INTO CARS(registered_time, photo_location) values('{datetime_now}', '{photo_location}');
        """
    datetime_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur = conn.cursor()
    cur.execute(query.format(datetime_now=datetime_now, photo_location=photo_location))
    conn.commit()
    return cur.lastrowid


def remove_old_records(conn, days=30):
    query = """
                DELETE FROM CARS WHERE date(registered_time) < date('{date_today}', '-{db_days} day');
                """
    date_today = datetime.now().strftime("%Y-%m-%d")
    cur = conn.cursor()
    num_rows = cur.execute(query.format(date_today=date_today, db_days=days))
    conn.commit()


def select_yesterday_records(conn):
    data = []
    query = """
        SELECT rowid, registered_time, photo_location FROM cars 
            WHERE is_processed = 0 and date(registered_time) = date('{date_today}', '-1 day');
    """
    cur = conn.cursor()
    date_today = datetime.now().strftime("%Y-%m-%d")
    cur.execute(query.format(date_today=date_today))

    rows = cur.fetchall()
    for row in rows:
        rowid, day, location = row
        data.append([rowid, day, location])
    return data


def update_processed_records(conn, id_str):
    query = """
            UPDATE CARS SET is_processed = 1 where rowid in ({id_str});
        """
    cur = conn.cursor()
    cur.execute(query.format(id_str=id_str))
    conn.commit()
