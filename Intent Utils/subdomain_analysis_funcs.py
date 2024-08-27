# @title Comb Func
from intent_utils import *
from pro_con_sa import pros_cons_seperator, pro_con_sentiment_assign, assign_final_sentiment
import re
import pandas as pd
import numpy as np
import os
from datetime import datetime
from comprative_config import *
from comparative_resolving_funcs import *
from competitor_analysis import *
from dataframe_preprocessing import *
from general_utils import zip_da_files

def subdomain_splitter(text, current_intent, subdomain_dict):
  """
  Subdomain Splitter: This function is used to split the text into subdomains based on the subdomain dictionary
  Args:
      text: str, text to be split
      current_intent: str, current intent for which subdomain is to be split
      subdomain_dict: dict, subdomain dictionary for the current intent
  Returns:
      pd.Series
  """
  current_subdomain_wds = subdomain_dict[current_intent]
  subdomain_identified=[]
  subdomain_re_wds=[]
  subdomain_wds_matched=[]
  for subdomain in current_subdomain_wds:
    for item in current_subdomain_wds[subdomain]:
        if item[:3]==r'^(?':
          a = re.findall(item , text, re.IGNORECASE)
        else:
          a = re.findall(r'\b'+item+r'\b',text, re.IGNORECASE)
        if a != [] and a!=['']:
          subdomain_wds_matched+=a
          subdomain_identified.append(subdomain)
          subdomain_re_wds.append(item)
  subdomain_identified=list(set(subdomain_identified))

  return pd.Series({'Subdomain':subdomain_identified, 'Subdomain_Matching_word':subdomain_wds_matched, 'Subdomain_re_word/s':subdomain_re_wds})


def combine_for_an_uid(new_cols, intent_dict, spw, subdomain_dict, intents):
  dic={}
  ##Seperate pros and cons to their seperate df and do subdomain splitting
  for i in new_cols.columns[1:]:
      temp_df=new_cols[['UID', i]].dropna().reset_index(drop=True)
      dic[i]=create_final_dataframe(temp_df, intent_dict, spw, i)
      if i=='Text':
        dic[i] = join_at_stopwords(dic[i].reset_index(drop=True))

      dic[i] = seperate_df(dic[i], intents)
      temp_subdomain_dfs={}
      final_dfs={}
      for j in intents:
        temp_subdomain_dfs[j] = dic[i][j]['Split'].apply(lambda text:subdomain_splitter(text, j, subdomain_dict))
        final_dfs[j]=pd.concat([dic[i][j], temp_subdomain_dfs[j]], axis=1)
      dic[i] = final_dfs

  ##Assign Sentiment for sub domains
  procondfs={}
  for i in dic:
    if i=='Text':
        continue
    procondfs[i]={}
    for j in intents:
      tdf=dic[i][j].set_index('UID').groupby('UID', group_keys=False).apply(lambda group: pro_con_sentiment_assign(group, i, intents = subdomain_dict[j].keys(), Intent_col='Subdomain'))
      procondfs[i][j]=tdf.reset_index()

  ##Merge pros and cons to get only dfs with keys as intents and each intent combination of pros and cons
  pro_con_df_fin={}
  for i in intents:
    lis = [k[i] for k in [procondfs['Pros'], procondfs['Cons']] if not k[i].empty]
    if len(lis)==0:
      continue
    pro_con_df_fin[i] = pd.concat(lis,axis=0).sort_values('UID').reset_index(drop=True)

  ##Merge for uids and give final sentiment
  final_pros_cons_comb={}
  for i in pro_con_df_fin:
    final_pros_cons_comb[i] = pro_con_df_fin[i].groupby('UID').agg(lambda x: assign_final_sentiment(x) if x.notna().any() else np.nan).reset_index()

  return final_pros_cons_comb

def uncombined_for_an_uid(new_cols, intent_dict, spw, subdomain_dict, intents):
    dic={}
    for i in new_cols.columns[1:]:
        senti_val=np.nan
        if i =='Pros':
            senti_val=1
        elif i=='Cons':
            senti_val=-1

        temp_df=new_cols[['UID', i]].dropna().reset_index(drop=True)
        dic[i]=create_final_dataframe(temp_df, intent_dict, spw, i)
        if i=='Text':
            dic[i] = join_at_stopwords(dic[i].reset_index(drop=True))

        dic[i] = seperate_df(dic[i], intents)
        temp_subdomain_dfs={}
        final_dfs={}

        for j in intents:
            temp_subdomain_dfs[j] = dic[i][j]['Split'].apply(lambda text:subdomain_splitter(text, j, subdomain_dict))
            final_dfs[j]=pd.concat([dic[i][j], temp_subdomain_dfs[j]], axis=1)
            final_dfs[j][j]=senti_val

        dic[i] = final_dfs

    pro_con_remaining = dic['Text']

    pros_cons_final={}
    for i in intents:
        lis=[dic[j][i] for j in dic if (not dic[j][i].empty and j!='Text')]
        if len(lis)==0:
          continue
        pros_cons_final[i]=pd.concat(lis, axis=0).sort_values('UID').reset_index(drop=True)
    #pros_cons_final['Price']

    return pros_cons_final, pro_con_remaining

def transform_df_for_sub_domain(row, main_intent, Subdomain_col, Subdomains):
  """
    This func transform given sentiment the dataframe to subdomain level
    Args:
        row: pd.Series
        main_intent: str, main intent for which sentiment is given
        Subdomain_col: str, column name for subdomains in the dataframe. ususally is 'Subdomain'
        Subdomains: list, list of subdomains for the main intent
    Returns:
        pd.Series
  """
  senti_val=row[main_intent]
  lis=row[Subdomain_col]
  senti={i:np.nan for i in Subdomains}

  for i in Subdomains:
    if i in lis:
      senti[i]=senti_val

  return pd.Series(senti)


def subdomain_analysis1(base_path, base_intent_path, model_name, og_df, func, onnx_path, tokenizer_path, intents, subdomain_dict, Intent_dict_path_and_info):
    """
    Subdomain Analysis: This function is used to perform subdomain analysis when the dataframe is not correct and Intent Analysis is to be run from scratch
    Args:
        base_path: str, Sentiment Analysis Folder Path
        base_intent_path: str, Intent Analysis Folder Path
        model_name: str, model name of product you wish to save with
        og_df: pd.DataFrame, Data you wish to perform subdomain analysis on
        func: function, function to be used for sentiment analysis, derived from main.py
        onnx_path: str, Model path
        tokenizer_path: str, tokenizer path
        intents: list, list of intents for the Product you want subdomain of
        subdomain_dict: dict, subdomain dictionary for the intents
    """
    intent_dict, spw = load_intent_dict(base_intent_path+Intent_dict_path_and_info[0], Intent_dict_path_and_info[1], Intent_dict_path_and_info[2])

    
    intent_dict2={}
    for i in intent_dict:
        k=[]
        temp=[]
        intent_dict[i].sort()
        for item in intent_dict[i]:
            # if item[:3]==r'^(?' or item[:3]==r'(?:':
            if (r'^(?' in item) or (item[:3]==r'(?:'):

                temp.append(item)
            else:
                k.append(item)
        temp.append(r'\b|\b'.join(k))
        intent_dict2[i]=temp
      

    ##Copy of Original df
    df= og_df.copy()

    df = process_df(df)

    # df = competitor_analyzer_dummy(base_path, base_intent_path, df, intent_dict, spw, func, onnx_path, tokenizer_path, msgs=True,dev_msgs=False)


    ##Pros and Cons
    pattern1 = r'\bpros\b ?:|\bCons\b ?:'
    pattern2 = r'\'.\''
    cond=df['Review Text'].str.contains(pattern1, case=False, regex=True) & df['Review Text'].str.contains(pattern2, case=False, regex=True)
    pros_cons = df[cond]
    df = df[~cond]
    # df=og_df[og_df.UID.isin(df.UID.to_list()+our_model_temp.UID.to_list()+parent_comp_temp.UID.to_list())].reset_index(drop=True)
    # df = process_df(df)

    if pros_cons.empty:
      print('No pros and cons')
    else:
      new_cols = pros_cons.apply(pros_cons_seperator, axis=1)
    #   print(new_cols)
      ##Combine for an UID
      final_pros_cons_comb = combine_for_an_uid(new_cols, intent_dict2, spw, subdomain_dict, intents)

      ##Uncombined for an UID
      pros_cons_final, pro_con_remaining = uncombined_for_an_uid(new_cols, intent_dict2, spw, subdomain_dict, intents)



    df = create_final_dataframe(df, intent_dict2, spw)
    df = join_at_stopwords(df.reset_index(drop=True))

    seperated_dfs=seperate_df(df, intents)

    temp_subdomain_dfs={}
    final_dfs={}

    for i in intents:
        temp_subdomain_dfs[i] = seperated_dfs[i]['Split'].apply(lambda text:subdomain_splitter(text, i, subdomain_dict))
        final_dfs[i]=pd.concat([seperated_dfs[i], temp_subdomain_dfs[i]], axis=1)

    final_dfs_to_thru_model={}
    if not pros_cons.empty:
      for i in intents:
          lis = [k[i] for k in [pro_con_remaining, final_dfs] if i in k and not k[i].empty]
          if len(lis)==0:
            continue
          final_dfs_to_thru_model[i] = pd.concat(lis,axis=0).sort_values('UID').reset_index(drop=True)
    else:
      final_dfs_to_thru_model = final_dfs


    for i in intents:
        final_dfs_to_thru_model[i].loc[:,i]=0

    for i in intents:

        if final_dfs_to_thru_model[i].empty:
            print('Empty df for ', i)
            continue

        print(f'filling sentiment for {i}...')
        onnx_path = onnx_path.format(i)
        tokenizer_path = tokenizer_path.format(i)

        final_dfs_to_thru_model[i].loc[:, i]=func(base_path, final_dfs_to_thru_model[i], i , 'Split', onnx_path=onnx_path, tokenizer_path=tokenizer_path )
        print('done')
    for i in intents:
        final_dfs_to_thru_model[i][i]=final_dfs_to_thru_model[i][i].apply(lambda x: x-1)

    comb_final_thru_model={}
    for i in intents:
      comb_final_thru_model[i]=final_dfs_to_thru_model[i].set_index('UID').apply(lambda group: transform_df_for_sub_domain(group, i, 'Subdomain', subdomain_dict[i]), axis=1).reset_index()

    if not pros_cons.empty:
      final_result_combined={}
      for i in intents:
          lis = [k[i] for k in [comb_final_thru_model, final_pros_cons_comb] if (i in k) and (not k[i].empty)]
          if len(lis)==0:
            continue
          final_result_combined[i] = pd.concat(lis,axis=0).sort_values('UID').reset_index(drop=True)
          final_result_combined[i] = final_result_combined[i].groupby('UID').agg(lambda x: assign_final_sentiment(x) if x.notna().any() else np.nan).reset_index()
    else:
      final_result_combined={}
      for i in intents:
          final_result_combined[i] = comb_final_thru_model[i].groupby('UID').agg(lambda x: assign_final_sentiment(x) if x.notna().any() else np.nan).reset_index()


    column_to_drop_after = 'Review Text'
    column_index = og_df.columns.get_loc(column_to_drop_after) + 1

    df2 = og_df.iloc[:, :column_index]

    current_date = datetime.now()
    day = str(current_date.day)
    short_month_name = current_date.strftime("%b")

    # Specify the path where you want to create the directory
    directory_path = base_intent_path+"Subdomain_files/"+model_name+f'/{day}_{short_month_name}'

    # Create the directory if it doesn't exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created at: {directory_path}")
    else:
        print(f"Directory already exists at: {directory_path}")


    combined_result={}

    for i in intents:
        combined_result[i] = pd.merge(df2, final_result_combined[i], how='right', on='UID')
        combined_result[i].to_excel(directory_path+f'/{i}_{model_name}_{day}_{short_month_name}_combined.xlsx', index=False)

    if not pros_cons.empty:
      final_result={}
      for i in intents:
          lis = [k[i] for k in [pros_cons_final, final_dfs_to_thru_model] if i in k and not k[i].empty]
          if len(lis)==0:
              continue
          final_result[i] = pd.concat(lis,axis=0).sort_values('UID').reset_index(drop=True)
          final_result[i].to_excel(directory_path + f'/{i}_{model_name}_{day}_{short_month_name}_uncombined.xlsx', index=False)
    else:
      for i in intents:
          final_dfs_to_thru_model[i].to_excel(directory_path + f'/{i}_{model_name}_{day}_{short_month_name}_uncombined.xlsx', index=False)

    uncomb_file_names_to_zip = [directory_path + f'/{i}_{model_name}_{day}_{short_month_name}_uncombined.xlsx' for i in intents]
    comb_file_names_to_zip = [directory_path + f'/{i}_{model_name}_{day}_{short_month_name}_combined.xlsx' for i in intents]

    zip_da_files(uncomb_file_names_to_zip, directory_path + f'/{model_name}_{day}_{short_month_name}_uncombined.zip')
    zip_da_files(comb_file_names_to_zip, directory_path + f'/{model_name}_{day}_{short_month_name}_combined.zip')
    print('All done!')
    print(f'Saved at: {directory_path}')

def subdomain_analysis2(base_path, base_intent_path, model_name, og_df, func, onnx_path, tokenizer_path, intents, subdomain_dict, Intent_dict_path_and_info):
    """
    Subdomain Analysis: This function is used to perform subdomain analysis when the dataframe is correct and Intent Analysis is not to be run from scratch and is already present,
    hence we can seperate the dataframe into  and perform subdomain analysis on the given data via intent dictionary subdomain sheet regexes

    Args:
        base_path: str, Sentiment Analysis Folder Path
        base_intent_path: str, Intent Analysis Folder Path
        model_name: str, model name you wish to save with
        og_df: pd.DataFrame, Data you wish to perform subdomain analysis on
        func: function, function to be used for sentiment analysis, derived from main.py
        onnx_path: str, Model path
        tokenizer_path: str, tokenizer path
        intents: list, list of intents for the Product you want subdomain of
        subdomain_dict: dict, subdomain dictionary for the intents
    """
    intent_dict, spw = load_intent_dict(base_intent_path+Intent_dict_path_and_info[0], Intent_dict_path_and_info[1], Intent_dict_path_and_info[2])


    intent_dict2={}
    for i in intent_dict:
        k=[]
        temp=[]
        intent_dict[i].sort()
        for item in intent_dict[i]:
            # if item[:3]==r'^(?' or item[:3]==r'(?:':
            if (r'^(?' in item) or (item[:3]==r'(?:'):

                temp.append(item)
            else:
                k.append(item)
        temp.append(r'\b|\b'.join(k))
        intent_dict2[i]=temp
        
    ##Copy of Original df
    main_df = og_df.copy()

    main_df = process_df(main_df)
    main_df.columns = correct_column_names(main_df)
    for intent in intents:
      # df = competitor_analyzer_dummy(base_path, base_intent_path, df, intent_dict, spw, func, onnx_path, tokenizer_path, msgs=True,dev_msgs=False)
      df = main_df[main_df[intent].notna()]

      ##Pros and Cons
      pattern1 = r'\bpros\b ?:|\bCons\b ?:'
      pattern2 = r'\'.\''
      cond=df['Review Text'].str.contains(pattern1, case=False, regex=True) & df['Review Text'].str.contains(pattern2, case=False, regex=True)
      pros_cons = df[cond]
      df = df[~cond]


      if pros_cons.empty:
        print('No pros and cons')
      else:
        new_cols = pros_cons.apply(pros_cons_seperator, axis=1)
        ##Combine for an UID
        final_pros_cons_comb = combine_for_an_uid(new_cols, intent_dict2, spw, subdomain_dict, intents)

        ##Uncombined for an UID
        pros_cons_final, pro_con_remaining = uncombined_for_an_uid(new_cols, intent_dict2, spw, subdomain_dict, intents)
      sub_intent_dict={intent:intent_dict2[intent]}

      df = create_final_dataframe(df, sub_intent_dict, spw)
      df = join_at_stopwords(df.reset_index(drop=True))

      seperated_dfs=seperate_df(df, [intent])

      temp_subdomain_dfs={}
      final_dfs={}

      ##
      temp_subdomain_dfs[intent] = seperated_dfs[intent]['Split'].apply(lambda text:subdomain_splitter(text, intent, subdomain_dict))

      final_dfs[intent]=pd.concat([seperated_dfs[intent], temp_subdomain_dfs[intent]], axis=1)

      final_dfs_to_thru_model={}
      if not pros_cons.empty:
        # for i in D:
        lis = [k[intent] for k in [pro_con_remaining, final_dfs] if intent in k and not k[intent].empty]
        if len(lis)==0:
          continue
        final_dfs_to_thru_model[intent] = pd.concat(lis,axis=0).sort_values('UID').reset_index(drop=True)
      else:
        final_dfs_to_thru_model = final_dfs


      # for i in intents:
      final_dfs_to_thru_model[intent].loc[:,intent]=0

      # for i in intents:
      if final_dfs_to_thru_model[intent].empty:
          print('Empty df for ', intent)
          continue

      print(f'filling sentiment for {intent}...')
      onnx_path = onnx_path.format(intent)
      tokenizer_path = tokenizer_path.format(intent)
      final_dfs_to_thru_model[intent].loc[:, intent]=func(base_path, final_dfs_to_thru_model[intent], intent , 'Split', onnx_path=onnx_path, tokenizer_path=tokenizer_path )
      print('done')

      # for i in intents:
      final_dfs_to_thru_model[intent][intent]=final_dfs_to_thru_model[intent][intent].apply(lambda x: x-1)

      comb_final_thru_model={}
      # for i in intents:
      comb_final_thru_model[intent]=final_dfs_to_thru_model[intent].set_index('UID').apply(lambda group: transform_df_for_sub_domain(group, intent, 'Subdomain', subdomain_dict[intent]), axis=1).reset_index()

      if not pros_cons.empty:
        final_result_combined={}
        # for i in intents:
        lis = [k[intent] for k in [comb_final_thru_model, final_pros_cons_comb] if (intent in k) and (not k[intent].empty)]
        if len(lis)==0:
          continue
        final_result_combined[intent] = pd.concat(lis,axis=0).sort_values('UID').reset_index(drop=True)
        final_result_combined[intent] = final_result_combined[intent].groupby('UID').agg(lambda x: assign_final_sentiment(x) if x.notna().any() else np.nan).reset_index()
      else:
        final_result_combined={}
        # for i in intents:
        final_result_combined[intent] = comb_final_thru_model[intent].groupby('UID').agg(lambda x: assign_final_sentiment(x) if x.notna().any() else np.nan).reset_index()


      column_to_drop_after = 'Review Text'
      column_index = og_df.columns.get_loc(column_to_drop_after) + 1

      df2 = og_df.iloc[:, :column_index]

      current_date = datetime.now()
      day = str(current_date.day)
      short_month_name = current_date.strftime("%b")

      # Specify the path where you want to create the directory
      directory_path = base_intent_path+"Subdomain_files/"+model_name+f'/{day}_{short_month_name}'

      # Create the directory if it doesn't exist
      if not os.path.exists(directory_path):
          os.makedirs(directory_path)
          print(f"Directory created at: {directory_path}")
      else:
          print(f"Directory already exists at: {directory_path}")


      combined_result={}

      # for i in intents:
      combined_result[intent] = pd.merge(df2, final_result_combined[intent], how='right', on='UID')
      combined_result[intent].to_excel(directory_path+f'/{intent}_{model_name}_{day}_{short_month_name}_combined.xlsx', index=False)

      if not pros_cons.empty:
        final_result={}
        # for i in intents:
        lis = [k[intent] for k in [pros_cons_final, final_dfs_to_thru_model] if intent in k and not k[intent].empty]
        if len(lis)==0:
            continue
        final_result[intent] = pd.concat(lis,axis=0).sort_values('UID').reset_index(drop=True)
        final_result[intent].to_excel(directory_path + f'/{intent}_{model_name}_{day}_{short_month_name}_uncombined.xlsx', index=False)
      else:
        # for i in intents:
        final_dfs_to_thru_model[intent].to_excel(directory_path + f'/{intent}_{model_name}_{day}_{short_month_name}_uncombined.xlsx', index=False)

    uncomb_file_names_to_zip = [directory_path + f'/{i}_{model_name}_{day}_{short_month_name}_uncombined.xlsx' for i in intents]
    comb_file_names_to_zip = [directory_path + f'/{i}_{model_name}_{day}_{short_month_name}_combined.xlsx' for i in intents]

    zip_da_files(uncomb_file_names_to_zip, directory_path + f'/{model_name}_{day}_{short_month_name}_uncombined.zip')
    zip_da_files(comb_file_names_to_zip, directory_path + f'/{model_name}_{day}_{short_month_name}_combined.zip')
    print(f'Saved at: {directory_path}')

def subdomain_analysis(base_path, base_intent_path, model_name, og_df, func, onnx_path, tokenizer_path, intents, Intent_dict_path_and_info, me_df=False):
    """
    Subdomain Analysis: This function is used to perform subdomain analysis on the given data via intent dictionary subdomain sheet regexes
    Args:
        base_path: str, Sentiment Analysis Folder Path
        base_intent_path: str, Intent Analysis Folder Path
        model_name: str, model name you wish to save with
        og_df: pd.DataFrame, Data you wish to perform subdomain analysis on
        func: function, function to be used for sentiment analysis, derived from main.py
        onnx_path: str, Model path
        tokenizer_path: str, tokenizer path
        intents: list, list of intents for the Product you want subdomain of
        Intent_dict_path: str, intent dictionary path
        me_df: bool, default False, if True then the dataframe is correct and no need to run Intent Analysis from scratch
    """
    ##Load Subdomain Dict
    subdomain_sheets={}
    subdomain_dict={}
    for i in intents:
        subdomain_dict[i]={}
        subdomain_sheets[i]=pd.read_excel(base_intent_path+Intent_dict_path_and_info[0], sheet_name=f'{i}')
        for j in subdomain_sheets[i]:
            subdomain_dict[i][j]=subdomain_sheets[i][j].dropna().tolist()
          
            k=[]
            temp=[]
            subdomain_dict[i][j].sort()
            for item in subdomain_dict[i][j]:
                if (r'^(?' in item) or (item[:3]==r'(?:'):
                    temp.append(item)
                else:
                    k.append(item)
            if len(k)>0: 
              temp.append(r'\b|\b'.join(k))
            subdomain_dict[i][j]=temp

    if not me_df:
        subdomain_analysis1(base_path, base_intent_path, model_name, og_df, func, onnx_path, tokenizer_path, intents, subdomain_dict, Intent_dict_path_and_info)
    else:
        subdomain_analysis2(base_path, base_intent_path, model_name, og_df, func, onnx_path, tokenizer_path, intents, subdomain_dict, Intent_dict_path_and_info)
