
def check_column_exists(conn, table_name, column_name):
    
    c = conn.cursor()

    c.execute("PRAGMA table_info('{}');".format(table_name))
    columns = c.fetchall()
    
    exsits = any(column[1] == column_name for column in columns)
    return exsits

def check_column_exists_add(conn, table_name, column_name, column_def=''):

    c = conn.cursor()
    
    if not check_column_exists(conn, table_name, column_name):
        c.execute("ALTER TABLE {} ADD COLUMN {} {};".format(table_name, column_name, column_def))
        conn.commit()
        print("Added column {} to table {}".format(column_name, table_name))
    else:
        print("Column {} already exists in table {}".format(column_name, table_name))