# from config import current_possible_intents as Mobile_intents
from intent_utils import *
import pandas as pd
import numpy as np
import re

#################### Pro Cons ####################
def pros_cons_seperator(row):
  """
    This function seperates the pros and cons from the review text, and keeps the remaining text in the 'Text' column
  """
  string=row['Review Text']
  UID=row['UID']
  pros_pat = r'\bPros\b\s?:([\w\W]*)'
  cons_pat = r'\bCons\b\s?:([\w\W]*)'
  pros=''
  cons=''
  rvwtxt=''
  # # strings=string.split('\'.\'')
  # strings=re.split(r"'\s?\.'|\.\s?'", string)
  strings=re.split(r"'\s?\.'|\.\s?'", string)

  for i in strings:
    # print(i)
    mat_pros=re.search(pros_pat, i, re.IGNORECASE)
    mat_cons=re.search(cons_pat, i, re.IGNORECASE)
    if mat_pros:
      pros+=mat_pros.group(1).strip()
    elif mat_cons:
      cons+=mat_cons.group(1).strip()
    else:
      rvwtxt+=i+', '
  if pros=='':
    pros=np.nan
  if cons=='':
    cons=np.nan
  if rvwtxt=='':
    rvwtxt=np.nan
  return pd.Series({'UID':UID, 'Pros':pros, 'Cons': cons, 'Text':rvwtxt})

def pro_con_sentiment_assign(group, pro_or_con, intents, Intent_col='Intent'):
  """
    This function assigns the sentiment to the pros and cons by counting the number of intents in the pros and cons individually
  """
  if pro_or_con=='Pros':
    senti_val=1
  else:
    senti_val=-1
  lis=[i for j in group[Intent_col] for i in j]
  # print(lis)
  senti={i:np.nan for i in intents}
  for i in intents:
    if i in lis:
      senti[i]=senti_val*lis.count(i)

  return pd.Series(senti)


def assign_final_sentiment(x):
    """
        This function assigns the final sentiment to pros-cons rows by combining the sentiments of the pros and cons and the result if >0 then +1, <0 then -1, else 0
    """
    if x.dropna().sum()>0:
        return 1
    elif x.dropna().sum()<0:
        return -1
    else:
        return 0



def get_sentences_where_intent_is_not_from_pro_cons(group, ints_found):
  """
    This function returns the sentences where the intents are not from pros and cons
  """
  
  def get_split_ids(lis, group):
    """
      This function returns the split ids of the sentences where the intents are present in corresponding UID of pros and cons
    """
    split_ids=[]
    list_of_intents_in_group = group.Intent.to_list()
    for i, id in zip(range(len(list_of_intents_in_group)) , group.Split_ID.to_list()):
      for j in list_of_intents_in_group[i]:
        if j not in lis:
          split_ids.append(id)
        else:
          list_of_intents_in_group[i].remove(j)
    group.Intent=list_of_intents_in_group
    return split_ids, group

  id=group['UID'].to_list()[0]
  cond=ints_found['UID']==id

  if id in ints_found['UID'].to_list():
    lis=ints_found[cond]['Intents'].to_list()[0]

    split_ids, group=get_split_ids(lis, group)
    group=group[group.Split_ID.isin(split_ids)]

  return group


def get_intents_of_pro_con_rows(row):
  """
    This function returns the intents of the row which has pros and cons
  """
  lis=row.dropna().index.to_list()
  lis.remove('UID')
  return pd.Series({'UID': row['UID'], 'Intents': lis })



def Pro_Con_Handler(base_path, pros_cons, intent_dict, spw, func, intents, msgs=True, onnx_path = None, tokenizer_path = None):
    """
        This function handles the pros and cons dataframe and returns the final dataframe with sentiment assigned to pros and cons and remaining text
    Args:
        base_path (str): path to the folder where the models are present
        pros_cons (pd.DataFrame): dataframe with pros and cons seperated from original df
        intent_dict (dict): dictionary with intents as keys and corresponding keywords as values
        spw (dict): dictionary with special words 
        func (function): function to be used for sentiment analysis please pass from main if you are using it or some other module, (Note: necessary to avoid circular imports)

    """

    new_cols = pros_cons.apply(pros_cons_seperator, axis=1)
    if msgs:
      print('dividing pros and cons and remaining text as seperate dataframes...')
    dic={}
    for i in new_cols.columns[1:]:
        temp_df=new_cols[['UID', i]].dropna().reset_index(drop=True)
        dic[i]=create_final_dataframe(temp_df, intent_dict, spw, i)
    if msgs:
      print('done')
      print('Assigning sentiment to pros and cons only...')
    procondfs={}
    for i in dic:
        if i=='Text':
            continue
        tdf=dic[i].set_index('UID').groupby('UID', group_keys=False).apply(lambda group: pro_con_sentiment_assign(group, i, intents=intents))
        procondfs[i]=tdf.reset_index()

    procondf = pd.concat(procondfs.values(), axis=0)
    
    procondf = procondf.groupby('UID').agg(lambda x: assign_final_sentiment(x) if x.notna().any() else np.nan).reset_index()

    ints_found=procondf.apply(get_intents_of_pro_con_rows, axis=1)
    if msgs:
      print('done')
      print('Assigning sentiment to remaining text by ml model...')
    remaining_after_pro_cons_df=dic['Text'].groupby('UID', group_keys=False).apply(lambda group: get_sentences_where_intent_is_not_from_pro_cons(group, ints_found)).reset_index(drop=True)
    remaining_after_pro_cons_df = join_at_stopwords(remaining_after_pro_cons_df)
    seperated_dfs=merge_like_sentences_without_senti(remaining_after_pro_cons_df, intents)

    # dummy numbers
    for i in intents:
        # seperated_dfs[i]=pd.read_excel(f"{i}_file.xlsx")
        seperated_dfs[i][i]=0

    for i in intents:
        if seperated_dfs[i].empty:
            if msgs:
                print('Empty df for ', i)
            continue
        if msgs:
          print(f'Current Ongoing: {i}...')
        
        seperated_dfs[i].loc[:, i] = func(base_path, seperated_dfs[i], i, onnx_path = onnx_path, tokenizer_path = tokenizer_path)
        if msgs:
          print('done')        
    for i in intents:
        seperated_dfs[i][i]=seperated_dfs[i][i].apply(lambda x: x-1)

    if msgs:
      print('Almost done with pros and cons!')

      print('Merging all results...')
    overall_df = pd.concat(seperated_dfs.values(), axis=0)
    overall_df.sort_values('UID', inplace=True)
    overall_df.drop('Review Text', axis=1, inplace=True)
    remaining_text_senti = overall_df.groupby('UID').agg(lambda x: x.dropna().iloc[0] if x.notna().any() else np.nan).reset_index()

    final_df=pd.concat([remaining_text_senti, procondf], axis=0)
    final_df = final_df.groupby('UID').agg(lambda x: x.dropna().iloc[0] if x.notna().any() else np.nan).reset_index()
  
    if msgs:
      print('Pros and Cons done', end='\n\n')
    
    return final_df