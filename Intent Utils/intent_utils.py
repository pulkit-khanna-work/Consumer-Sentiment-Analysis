import re
import pandas as pd
import zipfile
import os
import ast
import numpy as np
from config import mobile_human_error_dict, watch_human_error_dict, buds_human_error_dict, tablet_human_error_dict, stpwds, sep, to_keep_joined
from sklearn.metrics import precision_score, recall_score, f1_score
from datetime import datetime
from general_utils import zip_da_files
####################Utility Functions#######################
def correct_column_names(df, device='Mobile'):
    """
    This function is used to correct the column names of the given dataframe.
    Args:
        df (pd.DataFrame): Dataframe whose column names are to be corrected
    Returns:
        list: List of corrected column names
    """
    if device=='Mobile':
        human_error_dict=mobile_human_error_dict
    elif device=='Watch':
        human_error_dict=watch_human_error_dict
    elif device=='Buds':
        human_error_dict=buds_human_error_dict
    elif device=='Tablet':
        human_error_dict=tablet_human_error_dict


    cols=df.columns.to_list()

    for i in range(len(cols)):
        for j in human_error_dict:
            mat=re.search(j, cols[i], flags=re.IGNORECASE)
            if mat:
                cols[i]=human_error_dict[j]
    return cols

#################### Functions used to split sentences #######################
def split_sentence(sentence, separators=sep,stopwords=stpwds):
    """
    This function is used to split the sentence into parts and seperators.
    Args:
        sentence (str): Sentence to be splitted
        separators (list, optional): List of separators to be used. Defaults to seperators stored in config.
        stopwords (list, optional): List of stopwords to be used. Defaults to Stopwords stored in config.
    Returns:
        list: List of splitted parts
        list: List of seperators
    """

    # regex pattern for separators
    separator_pattern = '|'.join([re.escape(sepe) for sepe in separators])

    # Split the sentence using separators
    parts = re.split(separator_pattern, sentence)

    if stopwords!=None and stopwords!=[]:
      if separators!=None and separators!=[]:
        # regex pattern for stopwords with word boundaries
        stopword_pattern = '|'.join([rf"\b{re.escape(sw)}\b" for sw in stopwords])
        # Combine separator and stopword patterns for finding both separators and stopwords
        combined_pattern = f"({separator_pattern})|({stopword_pattern})"
      else:
        combined_pattern = stopword_pattern

    else:
      combined_pattern = separator_pattern

    # Find all occurrences of separators and stopwords in the sentence
    separating_words_iter = re.finditer(combined_pattern, sentence, flags=re.IGNORECASE)

    # Extract the separators and stopwords from the sentence
    separating_words = [match.group() for match in separating_words_iter]


    # Split the parts that contain stopwords
    final_parts = []
    for part in parts:
        if stopwords!=None and any(sw.lower() in part.lower() for sw in stopwords):
            split_parts = re.split(stopword_pattern, part, flags=re.IGNORECASE)
            final_parts.extend(split_parts)
        else:
            final_parts.append(part)

    # Remove leading/trailing whitespace and empty parts
    final_parts = [part.strip() for part in final_parts]
    if final_parts[-1]=='':
      final_parts.pop()
    #if there is no separator at the end of the sentence, add an empty string to the list of separators
    if len(final_parts)!=len(separating_words):
        separating_words.append('')
    return [final_parts, separating_words]


def load_intent_dict(path, mobile_sheet_name='MOBILE', special_words_sheet='Special_words_Mobile'):
    """
    This function is used to load the intent dictionary and special words dictionary from the given path.
    Args:
        path (str): Path to the intent dictionary
        mobile_sheet_name (str, optional): Sheet name of the intent dictionary excel file. Defaults to 'MOBILE'.
        special_words_sheet (str, optional): Sheet name of the special words dictionary excel file. Defaults to 'Special_words_Mobile'.

    """
    intent_dict_excel=pd.read_excel(path, sheet_name=mobile_sheet_name)
    intent_dict={}
    for i in intent_dict_excel:
        intent_dict[i]=intent_dict_excel[i].dropna().tolist()

    spw_excel=pd.read_excel(path, sheet_name=special_words_sheet)
    spw_excel.drop('Overlap', axis=1, inplace=True)
    
    spw_excel.dropna(axis=0, how='all', inplace=True)
    spw_excel.reset_index(drop=True, inplace=True)

    spw={}
    for i in range(len(spw_excel)):
        k=spw_excel.iloc[i,:].dropna()
        if len(k)>2:
            dic={}
            for word in k.iloc[2:].to_list():
                intent=word.split('(')[0]
                temp_var='('.join(word.split('(')[1:])
                j=temp_var[:-1].split(',')
                dic[intent]=j
        else:
            dic={}
        dic['Default']=k.iloc[1]
        spw[k.iloc[0]]=dic
    
    return intent_dict, spw

def create_final_dataframe(og_df, word_dict, spw, text_column='Review Text', ID_column='UID') -> pd.DataFrame:
    """
    This function is used to create the final dataframe with splitted sentences and their intents.
    Args:
        og_df (pd.DataFrame): Original dataframe
        word_dict (dict): Dictionary of intents
        spw (dict): Dictionary of special words
        text_column (str, optional): Name of the column containing the text. Defaults to 'Review Text'.
        ID_column (str, optional): Name of the column containing the unique ID. Defaults to 'UID'.
    Returns:
        pd.DataFrame: Final dataframe with splitted sentences and their intents
    """
    og_df=og_df[[ID_column, text_column]].dropna(how='any').reset_index(drop=True)
    sentences = og_df[text_column].to_list()

    df = pd.DataFrame(columns=['UID', 'Split_ID', 'Split', 'Splitted_at', 'Intent', 'Intent_words', 're_words'])
    current_row = 0
    current_col = 1

    # storage_keywords={i:[] for i in og_df.loc[:,'UID']}
    for sentence in sentences:
        #split the sentence
        last_intent=''
        prev_sep_word='<<prev>>'
        parts, separating_words = split_sentence(sentence)
        for part, sep_word in zip(parts, separating_words):
            matching_words=[]
            intents=[]
            keywdsfnd=[]
            #iterating every word
            #if part is empty or just a space, change last row's splitted at column with the current sep_word
            if part==' ' or part=='':
                df.loc[len(df)-1, 'Splitted_at']=repr(sep_word)
                continue
            
            for intent_name in word_dict:
              for item in word_dict[intent_name]:
                if item[:3]==r'^(?':
                  a = re.findall(item , part, flags=re.IGNORECASE)
                else:
                  a = re.findall(r'\b'+item+r'\b',part, flags=re.IGNORECASE)
                if a != [] and a!=['']:
                  matching_words+=a
                  intents.append(intent_name)
                  last_intent=intent_name
                  keywdsfnd.append(item)

            # intents=list(set(intents))

            #If no intent in regular dictionary is found, check them in special words dictionary
            #How about checking regularly from special dictionary?
            # if intents==[]:

            #iterate all special words
            for item in spw:
              if item[:3]==r'^(?':
                a = re.findall(item , part, flags=re.IGNORECASE)
              else:
                a = re.findall(r'\b'+item+r'\b',part, flags=re.IGNORECASE)

              #if ith special word is found
              if a != [] and a!=['']:
             
                found=0
                matching_words+=a
                #get all possible combination words associated with that ith special word
                intentwds=list(spw[item].keys())[:-1]

                #iterate through all associated words with that ith special word
                for i in intentwds:
                  # print(i)
                  # print(spw[item][i])
                  b=re.findall(r'\b'+'|'.join(spw[item][i])+r'\b',part, flags=re.IGNORECASE)

                  if b != [] and b!=['']:
                    matching_words+=b
                    intents.append(i)
                    if i!='Unclassified':
                      last_intent=i
                    found=1
                    keywdsfnd.append(item)
                    break
                #if no associated word found with ith special word then set the default intent
                if found==0:
                  intents.append(spw[item]['Default'])
                  if spw[item]['Default']!='Unclassified':
                    last_intent=spw[item]['Default']
                  keywdsfnd.append(item)
                  # break

            if (intents==[] or list(set(intents))==['Unclassified']) and last_intent!='' and part!='[This review was collected as part of a promotion':
              mat=re.match(r"\bthis\b|\bit\b|\bit's\b", part, flags=re.IGNORECASE)
              if mat:
                intents = [last_intent,'Ambigious']
                matching_words+=[mat.group()]

              else:
                # print(last_intent)
                if repr(prev_sep_word)!='\'\\n\'':# and repr(prev_sep_word)!='\'but\'':
                  intents = [last_intent,'Unclassified?']
                  matching_words+=['<<Sentiment Purpose>>']
                  last_intent=''
                else:
                  last_intent=''
            
            ###################Special Cases:###################
            #Only if 'Software' like intent is present in intents list
            temp_sent=','.join(list(word_dict.keys()))
            if 'softw' in temp_sent.lower() :
                #name of that software intent in intents list
                soft_intent=[i for i in list(word_dict.keys()) if 'softw' in i.lower()]
                #Easy to learn and use
                if part.strip()[:3].lower()=='use':
                # check in previous part of 'easy to' is present or not using regex
                  
                    if len(df)>0:
                      mat=re.search(r"\beasy to\b", df.loc[len(df)-1, 'Split'], flags=re.IGNORECASE)
                      if mat:
                          intents += soft_intent
                          matching_words+=[mat.group()]
                          keywdsfnd+=['easy to use special case']
                          last_intent=soft_intent[0]
            #####################################################

            #Adding the row
            df.loc[len(df)] = [og_df.iloc[current_row]['UID'], current_col, part, repr(sep_word), intents or ['Unclassified'], matching_words,keywdsfnd]
            current_col += 1
        current_row += 1
        current_col = 1
    return df

def sentence_merge_util(group)-> pd.Series:
    """Utility function to merge sentences within each group
    Args:
        group (pd.DataFrame): Group from a dataframe
    Returns:
        pd.Series: Merged sentence
    """
    merged_sentence = group['Split'][0] + ' ' + group['Splitted_at'].apply(ast.literal_eval)[0]+ ' ' + group['Split'][1]
    merged_sentence=merged_sentence.strip()
    # merged_sentence = ' '.join(group['Split'] + ' ' + group['Splitted_at'].apply(ast.literal_eval)).strip()
    # print(group)
    ints=[]
    intwrds=[]
    rewrds=[]

    for i in range(len(group)):
      ints+=group.loc[i, 'Intent']
      intwrds+=group.loc[i, 'Intent_words']
      rewrds+=group.loc[i, 're_words']
    # print(list(set(ints)), list(set(intwrds)))
    if ('Unclassified' in ints) and (len(set(ints))!=1):
      ints.remove('Unclassified')

    return pd.Series({'UID': group['UID'].iloc[0],'Split_ID': group['Split_ID'].iloc[-1] ,'Split': merged_sentence ,'Splitted_at': group['Splitted_at'].iloc[-1], 'Intent':list(set(ints)), 'Intent_words':list(set(intwrds)),  're_words':list(set(rewrds)) })

def join_at_stopwords(test_df, to_keep_joined=to_keep_joined):
  new_df=pd.DataFrame(columns=test_df.columns)
  to_join=pd.DataFrame(columns=test_df.columns)
  smdf=test_df.copy()
  smdf.dropna(subset=['Intent'], inplace=True)
  smdf.reset_index(drop=True, inplace=True)
  
  for i in range(len(smdf)-1):
    # print(i)
    to_join=pd.DataFrame(columns=smdf.columns)
    if smdf.loc[i,'Splitted_at'] in to_keep_joined:
      UID1=smdf.loc[i,'UID']
      UID2=smdf.loc[i+1,'UID']
      if UID1!=UID2:
        continue
      to_join.loc[len(to_join)]=smdf.loc[i,:]
      to_join.loc[len(to_join)]=smdf.loc[i+1,:]
      temp=to_join.groupby('UID', group_keys=False).apply(sentence_merge_util).reset_index(drop=True)
      # print(temp)
      smdf.loc[i+1]=temp.loc[0,:]
      smdf.drop(i, axis=0,inplace=True)
      new_df.loc[len(new_df)]=temp.loc[0,:]
    elif type(smdf.loc[i, 'Split']) == type(np.nan) or smdf.loc[i, 'Split'].strip()=='':
      smdf.drop(i, axis=0,inplace=True)
      # print()
      # print(test_df)
  smdf.reset_index(drop=True, inplace=True)
  return smdf

#################### Functions to be to just get seperated intents from above funcs without merging them #######################
def seperate_df_4_de(df, intents):
    """
    This function is used to seperate the intents from the dataframe for data-entry purposes. add overall intent for each UID.
    """
    seperated_dfs={}
    for i in intents:
        filtered_df = df[df['Intent'].apply(lambda x: i in x)]
        seperated_dfs[i]=filtered_df.reset_index()
        seperated_dfs[i].drop('index', axis=1,inplace=True)
        ##overall
        # Find the indices of the last occurrences of each unique ID
        last_occurrence_indices = seperated_dfs[i].drop_duplicates('UID', keep='last').index
        # print(last_occurrence_indices)
        nan_data = {
            'UID': seperated_dfs[i]['UID'].iloc[last_occurrence_indices],
            'Split_ID': np.nan,
            'Split': '~<<Overall>>~',
            'Splitted_at': np.nan,
            'Intent': np.nan,
            'Intent_words': np.nan,
        }
        nan_df = pd.DataFrame(nan_data)
        seperated_dfs[i]=pd.concat([seperated_dfs[i], nan_df], ignore_index=True)
        seperated_dfs[i].sort_values(by=['UID', 'Split_ID'], inplace=True)

        seperated_dfs[i][i]=np.nan
        seperated_dfs[i].to_excel(f"{i}_file.xlsx", index=False)
    
    # List of file names you want to zip (provide the actual file names)
    file_names_to_zip = [f"{i}_file.xlsx" for i in intents]

    # Create a zip file named 'zipped_files.zip'
    zip_file_name = 'intent_files.zip'

    # Open the zip file in write mode
    with zipfile.ZipFile(zip_file_name, 'w') as zipf:
        # Loop through the list of file names
        for file_name in file_names_to_zip:
            # Check if the file exists in the current directory
            if os.path.exists(file_name):
                # Add the file to the zip archive
                zipf.write(file_name)
            else:
                print(f"Warning: File '{file_name}' not found. Skipping...")
    return seperated_dfs

def seperate_df(df, intents):
    """
    This function is used to seperate the intents from the dataframe for any general purposes. just simple seperation
    """
    seperated_dfs={}
    for i in intents:
        filtered_df = df[df['Intent'].apply(lambda x: i in x)]
        seperated_dfs[i]=filtered_df.reset_index()
        seperated_dfs[i].drop('index', axis=1,inplace=True)
    return seperated_dfs

##############
############### Functions to be used when sentiment is given and merging same intent sentences #####################

def merge_like_sentences_without_senti2(df, intents):
    """
    This function is used to merge the sentences which belong to same intent.
    """
    seperated_dfs={}
    # vals={}
    for i in intents:
        filtered_df = df[df['Intent'].apply(lambda x: i in x)]
        seperated_dfs[i]=filtered_df.reset_index()
        seperated_dfs[i].drop('index', axis=1,inplace=True)

        # Group the DataFrame by 'id' and apply the custom function to merge sentences
        result_df = seperated_dfs[i].groupby('UID', group_keys=False).apply(sentence_merge_util3).reset_index(drop=True)
        if result_df.empty:
           result_df=pd.DataFrame(columns=['UID', 'Review Text', i])
        # using to_list as without it the index of the result_df[i] and vals[i] are not same
        result_df[i]=np.nan
        seperated_dfs[i]=result_df

    return seperated_dfs

def sentence_merge_util3(group):
    """Utility function to merge sentences within each group"""
    merged_sentence = ' '.join(group['Split'] + ' ' + group['Splitted_at'].apply(ast.literal_eval)).strip()
    
    # merged_sentence = ' '.join(group['Split'] + ' ' + group['Splitted_at'].apply(ast.literal_eval)).strip()
    # print(group)
    ints=[]
    intwrds=[]
    rewrds=[]
    
    for i in group.index:
      ints+=group.loc[i, 'Intent']
      intwrds+=group.loc[i, 'Intent_words']
      rewrds+=group.loc[i, 're_words']
    # print(list(set(ints)), list(set(intwrds)))
    if ('Unclassified' in ints) and (len(set(ints))!=1):
      ints.remove('Unclassified')

    return pd.Series({'UID': group['UID'].iloc[0],'Review Text': merged_sentence , 'Intent':list(set(ints)), 'Intent_words':list(set(intwrds)),  're_words':list(set(rewrds)) })


def sentence_merge_util2(group):
    """Utility function to merge sentences within each group"""
    merged_sentence = group['Split'][0] + ' ' + group['Splitted_at'].apply(ast.literal_eval)[0]+ ' ' + group['Split'][1]
    merged_sentence=merged_sentence.strip()
    # merged_sentence = ' '.join(group['Split'] + ' ' + group['Splitted_at'].apply(ast.literal_eval)).strip()
    # print(group)
    ints=[]
    intwrds=[]
    rewrds=[]

    for i in range(len(group)):
      ints+=group.loc[i, 'Intent']
      intwrds+=group.loc[i, 'Intent_words']
      rewrds+=group.loc[i, 're_words']
    # print(list(set(ints)), list(set(intwrds)))
    if ('Unclassified' in ints) and (len(set(ints))!=1):
      ints.remove('Unclassified')

    return pd.Series({'UID': group['UID'].iloc[0],'Split_ID': group['Split_ID'].iloc[-1] ,'Split': merged_sentence ,'Splitted_at': group['Splitted_at'].iloc[-1], 'Intent':list(set(ints)), 'Intent_words':list(set(intwrds)),  're_words':list(set(rewrds)) })

# Define a custom function to merge the sentences within each group
def merge_sentences(group):
    """Utility function to merge sentences within each group"""
    merged_sentence = ' '.join(group['Split'] + ' ' + group['Splitted_at'].apply(ast.literal_eval)).strip()
    return pd.Series({'UID': group['UID'].iloc[0], 'Review Text': merged_sentence})

def merge_like_sentences_without_senti(df, intents):
    """
    This function is used to merge the sentences which belong to same intent.
    """
    seperated_dfs={}
    # vals={}
    for i in intents:
        filtered_df = df[df['Intent'].apply(lambda x: i in x)]
        seperated_dfs[i]=filtered_df.reset_index()
        seperated_dfs[i].drop('index', axis=1,inplace=True)

        # Group the DataFrame by 'id' and apply the custom function to merge sentences
        result_df = seperated_dfs[i].groupby('UID', group_keys=False).apply(merge_sentences).reset_index(drop=True)
        if result_df.empty:
           result_df=pd.DataFrame(columns=['UID', 'Review Text', i])
        # using to_list as without it the index of the result_df[i] and vals[i] are not same
        result_df[i]=np.nan
        seperated_dfs[i]=result_df

    return seperated_dfs


def merge_like_sentences_without_senti(df, intents, og_df):
    """
    This function is used to merge the sentences which belong to same intent.
    """
    seperated_dfs={}
    vals={}
    for i in intents:
        filtered_df = df[df['Intent'].apply(lambda x: i in x)]
        seperated_dfs[i]=filtered_df.reset_index()
        seperated_dfs[i].drop('index', axis=1,inplace=True)

        # Group the DataFrame by 'id' and apply the custom function to merge sentences
        result_df = seperated_dfs[i].groupby('UID', group_keys=False).apply(merge_sentences).reset_index(drop=True)

        # Find the indices of the last occurrences of each unique ID
        last_occurrence_indices = seperated_dfs[i].drop_duplicates('UID', keep='last').index

        # Extract sentiment values from the original DataFrame using the indices and assign them to the result DataFrame
        vals[i]=og_df[og_df['UID'].isin(seperated_dfs[i]['UID'].iloc[last_occurrence_indices])][i]

        # using to_list as without it the index of the result_df[i] and vals[i] are not same
        result_df[i]=vals[i].to_list()
        seperated_dfs[i]=result_df
        
    return seperated_dfs


def seperate_nan_vals(seperated_dfs, save_path):
    """
    This function is used to seperate the NaN values from the dataframe and save them in a seperate file.
    """
    nan_dfs={}
    for key in seperated_dfs.keys():
        nan_dfs[key]=seperated_dfs[key][seperated_dfs[key].isna().any(axis=1)]
        nan_dfs[key][key]=1
        seperated_dfs[key]=seperated_dfs[key].dropna()

        seperated_dfs[key].to_csv(save_path+f"{key}_file.csv", index=False)

    nan_df=pd.concat(nan_dfs.values(), axis=0).reset_index(drop=True)
    nan_df.to_excel(save_path+'NAN_VALS.xlsx', index=False)
    return nan_df



#################### Functions to remerge the splitted intent files #####################


def seperate_overall_n_review_title(dfs):
    overall_rows={}
    review_title={}
    for i in dfs:
        df=dfs[i]
        filtered_df = df.loc[df['Split'].str.contains('~<<Overall>>~')]
        overall_rows[i]=filtered_df
        df.drop(df[df['Split'].str.contains('~<<Overall>>~')].index, inplace=True)

        review_df = df.loc[df['Splitted_at'].str.contains("<<TB4>>")]

        review_title[i]=review_df
        df.drop(df[df['Splitted_at'].str.contains('<<TB4>>')].index, inplace=True)

        dfs[i]=df
    #Make one df out of all dfs
    review_df=pd.concat(review_title.values(), axis=0).reset_index(drop=True)
    review_df=review_df.groupby('UID', group_keys=False).first()
    review_df.sort_values('UID',inplace=True)

    overall_df = pd.concat(overall_rows.values(), axis=0)
    overall_df.sort_values('UID', inplace=True)
    overall_df.reset_index(drop=True,inplace=True)

    merged_df = overall_df.groupby('UID', group_keys=False).agg(lambda x: x.dropna().iloc[0] if x.notna().any() else np.nan).reset_index()
    overall_df=merged_df.drop(['Split', 'Split_ID', 'Splitted_at', 'Intent', 'Intent_words'], axis=1)
    return dfs, overall_df, review_df, merged_df

#################### Function to check % intent cols similarity between 2 df #####################
def check_against_df(base_intent_path, me_df, de_df, intents, load_intent_of='Mobile', Correct_Names=True, check_weird_senti_vals=True):
    """
    This function checks the given dataframe against the dataframe that is given to it, the scores coming out are assuming 2nd dataframe is 100% correct and signifies the accuracy of the first dataframe
    Args:
        me_df (pd.DataFrame): Dataframe to be checked(Acronyms: Machine Evaluated)
        de_df (pd.DataFrame): Dataframe to be checked against(Acronyms: Data Entry Estimated)
        intents (list): List of intents to be checked
        Correct_Names (bool, optional): Whether to correct the column names of the given dataframes. Defaults to True.
    Returns:

    """
    ###Below piece is used for naming only
    current_date = datetime.now()
    day = str(current_date.day)
    short_month_name = current_date.strftime("%b")
    ###

    if load_intent_of=='Mobile':
        intent_dict, spw = load_intent_dict(base_intent_path+'Intent_Dictionaries/Mobile_Intent_dict.xlsx')
    elif load_intent_of=='Watch':
        intent_dict, spw = load_intent_dict(base_intent_path+'Intent_Dictionaries/Watch_Intent_dict.xlsx', mobile_sheet_name='Watches', special_words_sheet='Special_words_Watch')
    elif load_intent_of =='Buds':
        intent_dict, spw = load_intent_dict(base_intent_path+'Intent_Dictionaries/buds_ANC_Intent_dict.xlsx', 'buds','Special_words_Buds' )
    elif load_intent_of =='Tablet':
        intent_dict, spw = load_intent_dict(base_intent_path+'Intent_Dictionaries/Tablet_Intent_dict.xlsx', 'Tablet','Special_words_Tablet'  )
    
    if Correct_Names:
        if load_intent_of=='Mobile':
            me_df.columns=correct_column_names(me_df)
            de_df.columns=correct_column_names(de_df)
        elif load_intent_of=='Watch':
            de_df.columns=correct_column_names(de_df, 'Watch')
            me_df.columns=correct_column_names(me_df, 'Watch')
        elif load_intent_of=='Buds':
            de_df.columns=correct_column_names(de_df, 'Buds')
            me_df.columns=correct_column_names(me_df, 'Buds')
        elif load_intent_of=='Tablet':
            de_df.columns=correct_column_names(de_df, 'Tablet')
            me_df.columns=correct_column_names(me_df, 'Tablet')
        
    cols=intents
    vals=[0, 1, -1, np.nan]


    def check_weird_senti_values(df):
        """
        This function checks if there are any weird values in the sentiment column of the given dataframe
        Args:
            df (pd.DataFrame): Dataframe to be checked
        """
        weird_intents=[]

        for i in cols:
            f_cols = df[~df[i].isin(vals)]
            if not f_cols.empty:
                weird_intents.append(i)
                print(i, 'has weird values in it, please check the following rows: ')
                print(df[~df[i].isin(vals)][['UID', i]])
                df=df[df[i].isin(vals)]
        return df
    if check_weird_senti_vals:
        print('Checking if there is any weird sentiment values in the Human evaluated dataframe')
        de_df=check_weird_senti_values(de_df)
        print('Checking if there is any weird sentiment values in the machine evaluated dataframe')
        me_df=check_weird_senti_values(me_df)

    me_df=me_df[me_df.UID.isin(de_df.UID.to_list())].reset_index(drop=True)
    de_df=de_df[de_df.UID.isin(me_df.UID.to_list())].reset_index(drop=True)

    og_me_df=me_df.copy()
    og_de_df=de_df.copy()
    

    print('Accuracy of the dataframe(Overall Sentiment + Intent): ')
    # Calculate accuracy for each column
    accuracy_dict = {}
    total_rows = de_df.shape[0]

    for col in cols:
        col1 = col
        col2 = col
        de_df[col1]=de_df[col1].astype(float)

        me_df[col2]=me_df[col2].astype(float)
        accurate_count = ((pd.isna(de_df[col1]) & pd.isna(me_df[col2])) | (de_df[col1] == me_df[col2])).sum()
        accuracy = accurate_count / total_rows * 100

        accuracy_dict[col] = accuracy

    # Print accuracy results
    for col, accuracy in accuracy_dict.items():
        print(f"Accuracy for {col}: {accuracy:.2f}%")
    print('wish to store mismatched rows? (y/n)')
    while True:
        try:
            choice=input()
            if choice not in ['y','n']:
                raise Exception
            break
        except:
            print("Please enter a valid choice")
    if choice=='y':
        print('Storing mismatched rows...')
        mismatched={}
        total=0
        for col in cols:
            col1 = col
            col2 = col
            de_mismatched=de_df[~((pd.isna(de_df[col1]) & pd.isna(me_df[col2])) | (de_df[col1] == me_df[col2]))][['UID', col]]
            if 'Review Title' in me_df.columns:
               cols_to_consider=['UID', 'Review Text', 'Review Title', col]
            else:
                cols_to_consider=['UID', 'Review Text', col]
            mismatches=me_df[~((pd.isna(de_df[col1]) & pd.isna(me_df[col2])) | (de_df[col1] == me_df[col2]))][cols_to_consider]
            df=pd.concat([mismatches.set_index('UID'), de_mismatched.set_index('UID')], axis=1).reset_index()
            temp_df=df.copy()
            columns=cols_to_consider[:-1]+[f'{col}_Machine_identified', f'{col}_Data_Entry_identified']
            df.columns=columns
            total+=len(df)
            # df['Review Title']=df['Review Title'].fillna('<<NULL>>')
            # df['Review Text'] = df['Review Title'].astype(str) + '<<TB4>>' + df['Review Text']
            # df.drop('Review Title', axis=1, inplace=True)
            temp_df=df.rename({'Review Text':'Full Review Text'}, axis=1).copy()

            df=create_final_dataframe(df, intent_dict, spw)
            df = join_at_stopwords(df)
            # intents=['Price', 'Build and Design', 'Screen', 'Camera',	'Battery',	'Performance_Processor',	'Software',	'NETWORKING',	'Other_Specs']
            seperated_dfs=merge_like_sentences_without_senti2(df, intents)

            df=seperated_dfs[col]
            df.drop(col,axis=1,inplace=True)

            # print(df)
            # break
            df.rename({'Review Text':'Relevant Review Text'}, axis=1, inplace=True)
            df=pd.concat([df.set_index('UID'), temp_df.set_index('UID')], axis=1).reset_index()
            df.to_excel(f'{col}_mismatched.xlsx', index=False)
            mismatched[col]=df
        print(f'Total mismatched rows: {total}')

        print('Zipping the files...')
        # List of file names you want to zip (provide the actual file names)
        file_names_to_zip = [f"{i}_mismatched.xlsx" for i in cols]

        # Create a zip file named 'zipped_files.zip'
        zip_file_name = f'Mismatched_files_{day}_{short_month_name}.zip'

        zip_da_files(file_names_to_zip, zip_file_name)

        # Move the zip file to the desired location (optional)
        # You can download it from the left sidebar
        # Or save it to Google Drive
        print('done')


####Intent only accuracy check
    print('Want to check the accuracy of the Intents only? (y/n)')
    while True:
        try:
            choice=input()
            if choice not in ['y','n']:
                raise Exception
            break
        except:
            print("Please enter a valid choice")
    if choice=='y':
        print('Checking the accuracy of the Intents only...')
        val=[0, 1, -1]

        for i in cols:
            me_df.loc[me_df[i].isin(val), i]=1
            me_df[i].fillna(0, inplace=True)

            de_df.loc[de_df[i].isin(val), i]=1
            de_df[i].fillna(0, inplace=True)


        print('Accuracy of the dataframe(Intents only): ')
        # Calculate accuracy for each column
        accuracy_dict = {}
        total_rows = de_df.shape[0]

        for col in cols:
            col1 = col
            col2 = col

            Tru=de_df[col1].values
            pred=me_df[col2].values


            #*************************Intents scores modified******************************#
            #precision
            
            precision_overall = precision_score(Tru, pred, average = 'weighted')*100
            precision = precision_score(Tru, pred)*100
            
            
            #-----------------------------------------------------
            #recall

            recall = recall_score(Tru, pred)*100
            recall_overall = recall_score(Tru, pred, average = 'weighted') *100
            


            #-----------------------------------------------------
            #f1score

            f1_overall = f1_score(Tru, pred, average= 'weighted')*100
            f1 = f1_score(Tru, pred)*100
    

            #-----------------------------------------------------
            #accuracy
            matched = de_df[de_df[col] == me_df[col]]
            total_rows = de_df.shape[0]

            # matched_1s  = matched[col].value_counts().get(1,0)
            # de_df_1s    = de_df[col].value_counts().get(1,0)
            try:
                accuracy_1s = matched[col].value_counts().get(1,0)/de_df[col].value_counts().get(1,0)
            except ZeroDivisionError:
                accuracy_1s = 0

            try:
                accuracy_0s = matched[col].value_counts().get(0,0)/de_df[col].value_counts().get(0,0)
            except ZeroDivisionError:
                accuracy_0s = 0


            accurate_count = (de_df[col] == me_df[col]).sum()
            accuracy = accurate_count / total_rows * 100
            

            accuracy_dict[col] = [accuracy, precision_overall, precision, recall_overall, recall, f1_overall, f1, accuracy_1s, accuracy_0s]
            
            
            

        # Print accuracy results
        for col, scores in accuracy_dict.items():
            accuracy, precision_overall, precision, recall_overall,recall, f1_overall, f1, accuracy_1s, accuracy_0s = scores
            print('--------------------------------------------------------------')
            print(f'|\033[1m Accuracy of {col}: {accuracy:.2f}% \033[0m |')
            print(f'|Accuracy Score for 1 only:{accuracy_1s:.2f}%')
            print(f'|Accuracy Score for 0 only:{accuracy_0s:.2f}%')
            print('--------------------------------------------------------------')
            print(f'|Precision:{precision_overall:.2f}%')
            print(f'|Precision for Intents only(1): {precision:.2f}%')
            print(f'|Recall:{recall_overall:.2f}%')
            print(f'|Recall for Intents only(1):{recall:.2f}%')
            print(f'|F1-Score:{f1_overall:.2f}%')
            print(f'|F1-Score for Intents only:{f1:.2f}%')
            
            
            print('\n')
        
        print('wanna store mismatched rows for Intents only? (y/n)')
        while True:
            try:
                choice=input()
                if choice not in ['y','n']:
                    raise Exception
                break
            except:
                print("Please enter a valid choice")

        if choice=='y':
            mismatched={}
            total=0
            for col in cols:
                col1 = col
                col2 = col
                de_mismatched=de_df[~(de_df[col]==me_df[col])][['UID', col]]
                if 'Review Title' in me_df.columns:
                    cols_to_consider=['UID', 'Review Text', 'Review Title', col]
                else:
                    cols_to_consider=['UID', 'Review Text', col]

                mismatches=me_df[~(de_df[col]==me_df[col])][cols_to_consider]
                df=pd.concat([mismatches.set_index('UID'), de_mismatched.set_index('UID')], axis=1).reset_index()
                temp_df=df.copy()
                columns=cols_to_consider[:-1]+[f'{col}_Machine_identified', f'{col}_Data_Entry_identified']
                df.columns=columns
                total+=len(df)
                
                temp_df=df.rename({'Review Text':'Full Review Text'}, axis=1).copy()
                
                temp_dict={col: intent_dict[col]}

                df=create_final_dataframe(df, temp_dict, spw)
                df = join_at_stopwords(df)

                # intents=['Price', 'Build and Design', 'Screen', 'Camera',	'Battery',	'Performance_Processor',	'Software',	'NETWORKING',	'Other_Specs']
                seperated_dfs=merge_like_sentences_without_senti2(df, intents)

                df=seperated_dfs[col]
                df.drop(col,axis=1,inplace=True)

                # print(df)
                # break
                df.rename({'Review Text':'Relevant Review Text'}, axis=1, inplace=True)
                df=pd.concat([df.set_index('UID'), temp_df.set_index('UID')], axis=1).reset_index()
                df.to_excel(f'{col}_intent_mismatched.xlsx', index=False)
                mismatched[col]=df

            print(f'Total mismatched rows: {total}')
            print('Zipping the files...')

            # List of file names you want to zip (provide the actual file names)
            file_names_to_zip = [f"{i}_intent_mismatched.xlsx" for i in cols]

            # Create a zip file named 'zipped_files.zip'
            zip_file_name = f'Intent_mismatched_{day}_{short_month_name}.zip'

            zip_da_files(file_names_to_zip, zip_file_name)

            # Move the zip file to the desired location (optional)
            # You can download it from the left sidebar
            # Or save it to Google Drive

####Sentiment only accuracy check
    
    print('Want to check the accuracy of the Sentiment only? (y/n)')
    while True:
        try:
            choice=input()
            if choice not in ['y','n']:
                raise Exception
            break
        except:
            print("Please enter a valid choice")
    if choice=='y':
        print('Checking the accuracy of the Sentiment only...')
        # val=[0, 1, -1]
        for col in cols:

            df1=og_de_df[['UID', col]].dropna().reset_index(drop=True)
            df2=og_me_df[['UID', col]].dropna().reset_index(drop=True)


            df1=df1[df1.UID.isin(df2.UID)].reset_index(drop=True)
            df2=df2[df2.UID.isin(df1.UID)].reset_index(drop=True)

            Tru=df1[col].values
            pred=df2[col].values

            precision = precision_score(Tru, pred, average=None)
            recall = recall_score(Tru, pred, average=None)
            f1 = f1_score(Tru, pred, average=None)

            # print(col)
            # print(len(df1))
            print(df1[col].value_counts())

            print(f"Accuracy: {(df1[col]==df2[col]).sum()/len(df1):.4f}")

            for i in range(len(precision)):
                print(f"Precision for class {i-1}: {precision[i]:.4f}")
                print(f"Recall for class {i-1}: {recall[i]:.4f}")
                print(f"F1-score for class {i-1}: {f1[i]:.4f}")
                print()
            print()
    else:
        print('done')
        return

    print('wanna store mismatched rows for Sentiment only? (y/n)')
    while True:
        try:
            choice=input()
            if choice not in ['y','n']:
                raise Exception
            break
        except:
            print("Please enter a valid choice")

    if choice=='y':
        mismatched={}
        total=0
        for col in cols:
            
            df1=og_de_df[['UID', col]].dropna().reset_index(drop=True)
            df2=og_me_df[['UID', col]].dropna().reset_index(drop=True)
            
            df1=og_de_df[og_de_df['UID'].isin(df1.UID.to_list())].reset_index(drop=True)
            df2=og_me_df[og_me_df['UID'].isin(df2.UID.to_list())].reset_index(drop=True)

            df1=df1[df1.UID.isin(df2.UID)].reset_index(drop=True)
            df2=df2[df2.UID.isin(df1.UID)].reset_index(drop=True)

            de_mismatched=df1[~(df1[col]==df2[col])][['UID', col]]
            if 'Review Title' in df2.columns:
               cols_to_consider=['UID', 'Review Text', 'Review Title', col]
            else:
                cols_to_consider=['UID', 'Review Text', col]

            mismatches=df2[~(df1[col]==df2[col])][cols_to_consider]
            df=pd.concat([mismatches.set_index('UID'), de_mismatched.set_index('UID')], axis=1).reset_index()
            temp_df=df.copy()
            columns=cols_to_consider[:-1]+[f'{col}_Machine_identified', f'{col}_Data_Entry_identified']
            df.columns=columns
            total+=len(df)
            
            temp_df=df.rename({'Review Text':'Full Review Text'}, axis=1).copy()
            
            temp_dict={col: intent_dict[col]}

            df=create_final_dataframe(df, temp_dict, spw)
            df = join_at_stopwords(df)

            # intents=['Price', 'Build and Design', 'Screen', 'Camera',	'Battery',	'Performance_Processor',	'Software',	'NETWORKING',	'Other_Specs']
            seperated_dfs=merge_like_sentences_without_senti2(df, intents)

            df=seperated_dfs[col]
            df.drop(col,axis=1,inplace=True)

            # print(df)
            # break
            df.rename({'Review Text':'Relevant Review Text'}, axis=1, inplace=True)
            df=pd.concat([df.set_index('UID'), temp_df.set_index('UID')], axis=1).reset_index()
            df.to_excel(f'{col}_sentiment_mismatched.xlsx', index=False)
            mismatched[col]=df

        print(f'Total mismatched rows: {total}')
        print('Zipping the files...')

        # List of file names you want to zip (provide the actual file names)
        file_names_to_zip = [f"{i}_sentiment_mismatched.xlsx" for i in cols]

        # Create a zip file named 'zipped_files.zip'
        zip_file_name = f'Sentiment_Mismatched_{day}_{short_month_name}.zip'

        zip_da_files(file_names_to_zip, zip_file_name)

        # Move the zip file to the desired location (optional)
        # You can download it from the left sidebar
        # Or save it to Google Drive
    else:
        print('done')
        return

