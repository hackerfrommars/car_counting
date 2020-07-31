import os
import shutil
from datetime import date, timedelta
from utils import remove_old_records, create_connection

FILE_DAYS = 7


def main():
    conn = create_connection()
    with conn:
        remove_old_records(conn)
        print(f"removed DB records")

    today = date.today()
    days_list = {}
    for i in range(FILE_DAYS):
        day = today - timedelta(days=i)
        days_list[str(day.strftime("%Y_%m_%d"))] = True

    d = './out'
    all_days = [o for o in os.listdir(d) if os.path.isdir(os.path.join(d, o))]
    for day in all_days:
        if not days_list.get(day, False):
            shutil.rmtree(f"./out/{day}")
            print(f"removed directory {day}")


if __name__ == "__main__":
    main()
