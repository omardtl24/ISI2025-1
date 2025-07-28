import pandas as pd # type: ignore

def parse_table(table_str, index = False, target = False, columns = False, target_name = None):
    """
    Parse a table string into a pandas DataFrame.
    
    Args:
        table_str (str): The table string to parse.
        index (bool): Whether to set the first column as the index. Default is False.
        target (bool): Whether to set the last column as the target. Default is False.
        columns (bool): Whether to set the first row as the column names. Default is False.
    
    Returns:
        pd.DataFrame: The parsed DataFrame.
    """
    # Split the string into rows
    rows = table_str.strip().split('\n')
    
    # Split each row into columns
    data = [row.split() for row in rows]
    
    # Create a DataFrame from the data
    if columns:
        df = pd.DataFrame(data[1:], columns=data[0])
    else:
        df = pd.DataFrame(data)

    # Set the first column as the index if specified
    if index:
        df.set_index(df.columns[0], inplace=True)
    
    # Set the last column as the target if specified
    if target and target_name not in df.columns:
        df[target_name] = df[df.columns[-1]]
        df.drop(df.columns[-1], axis=1, inplace=True)
    
    return df