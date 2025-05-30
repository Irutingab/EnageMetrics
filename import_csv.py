import csv
from db_connection import DatabaseConnection

CSV_FILE = "c:\\Users\\RAISSA\\Documents\\EnageMetrics\\parent_engagement_mock_data.csv"

def import_data():
    with open(CSV_FILE, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = [row for row in reader]

    insert_query = """
        INSERT INTO parent_engagement (
            Student_ID, Student_Name, Grade_Term1, Grade_Term2, Grade_Term3,
            Logins_Term1, Logins_Term2, Logins_Term3,
            Messages_Term1, Messages_Term2, Messages_Term3,
            Attendance_Term1, Attendance_Term2, Attendance_Term3,
            Total_Logins, Total_Messages, Avg_Grade, Avg_Attendance
        ) VALUES (
            %(Student_ID)s, %(Student_Name)s, %(Grade_Term1)s, %(Grade_Term2)s, %(Grade_Term3)s,
            %(Logins_Term1)s, %(Logins_Term2)s, %(Logins_Term3)s,
            %(Messages_Term1)s, %(Messages_Term2)s, %(Messages_Term3)s,
            %(Attendance_Term1)s, %(Attendance_Term2)s, %(Attendance_Term3)s,
            %(Total_Logins)s, %(Total_Messages)s, %(Avg_Grade)s, %(Avg_Attendance)s
        )
        ON CONFLICT (Student_ID) DO NOTHING;
    """

    with DatabaseConnection() as conn:
        with conn.cursor() as cur:
            for row in rows:
                # Convert numeric fields
                for key in row:
                    if key not in ['Student_ID', 'Student_Name']:
                        if row[key] == '':
                            row[key] = None
                        else:
                            try:
                                if '.' in row[key]:
                                    row[key] = float(row[key])
                                else:
                                    row[key] = int(row[key])
                            except ValueError:
                                row[key] = None
                cur.execute(insert_query, row)
        conn.commit()

if __name__ == "__main__":
    import_data()
