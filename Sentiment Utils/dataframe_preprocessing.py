import pandas as pd

def process_df(df):
    """
    This function is used to process the dataframe before feeding it to the model.
    It replaces the null values in the Review Title column with <<NULL>> and then concatenates the Review Title and Review Text columns.
    It then drops the Review Title column.
    It also replaces multiple dots, question marks and exclamation marks with a single dot, question mark and exclamation mark respectively.
    Args:
        df (pd.DataFrame): Dataframe to be processed
    Returns:
        pd.DataFrame: Processed dataframe

    """

    if "Review Title" in df.columns:
        df['Review Title']=df['Review Title'].fillna('<<NULL>>')
        df[df['Review Title']=='<<NULL>>']['Review Title']
        df['Review Text'] = df['Review Title'].astype(str) + '<<TB4>>' + df['Review Text']
        df.drop('Review Title', axis=1, inplace=True)

    df['Review Text']=df['Review Text'].str.replace(r'[\.]+', '.', regex=True)
    df['Review Text']=df['Review Text'].str.replace(r'[\?]+', '?', regex=True)
    df['Review Text']=df['Review Text'].str.replace(r'[\!]+', '!', regex=True)
    
    return df
