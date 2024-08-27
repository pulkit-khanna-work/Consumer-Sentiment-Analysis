import pandas as pd
import re
import ast
from intent_utils import split_sentence, merge_sentences, join_at_stopwords, create_final_dataframe
# from config import current_possible_intents as Mobile_intents
from comprative_config import *
from comparative_resolving_funcs import *
import numpy as np

def load_comparative_dict(path, Comb_sheet='Combined'):
    """
    This function is used to load the comparative dictionary from the given path.
    Args:
        path (str): Path to the comparative dictionary
        brand_sheet (str, optional): Sheet name where the brands kewyords are present. Defaults to 'brand'.
        Comb_sheet (str, optional): Sheet name of the combined keywords sheet. Defaults to 'Combined'.

    """

    combine_dict_df=pd.read_excel(path, sheet_name=Comb_sheet)
    model_dict={}
    company_dict={}
    for i in combine_dict_df:
        brand_re=[]
        for item in combine_dict_df[i].dropna().tolist():
            model_name, model_re = item.strip().split('<<>>')
            model_dict[model_name]=[model_re]
            brand_re.append(model_re)
        company_dict[i]=brand_re

    return model_dict, company_dict

def competitors_splitter(og_df, word_dict:dict, company_dict, text_column='Review Text', ID_column='UID', model_col='Model Name'):
    """
    This function is used to split the sentences in the given dataframe and find the competitor mentioned in the sentence.
    Args:
        og_df (pd.DataFrame): Dataframe to be used for analysis
        word_dict (dict): Dictionary containing the keywords for each intent
        company_dict (dict): Dictionary containing the keywords for overall brand
        text_column (str, optional): Name of the column where the text is present. Defaults to 'Review Text'.
        ID_column (str, optional): Name of the column where the unique ID is present. Defaults to 'UID'.
        model_col (str, optional): Name of the column where the model name is present. Defaults to 'Model Name'.
    Returns:
        pd.DataFrame: Dataframe containing the splitted sentences and the competitor mentioned in the sentence.
    """

    og_df=og_df[[ID_column, text_column, model_col]].dropna(how='any').reset_index(drop=True)
    sentences = og_df[text_column].to_list()
    df = pd.DataFrame(columns=['UID', 'Split_ID', model_col, 'Split', 'Splitted_at', 'competitor_mentioned', 'competitor_word', 'regex_word'])
    current_row = 0
    current_col = 1
    for sentence in sentences:
        #split the sentence
        last_intent=''
        parts, separating_words = split_sentence(sentence, separators=comp_sep, stopwords=None)
        for part, sep_word in zip(parts, separating_words):
            matching_words=[]
            intents=[]
            keywdsfnd=[]
            model_name = og_df.iloc[current_row][model_col]
            
            if part==' ' or part=='':
                df.loc[len(df)-1, 'Splitted_at']=repr(sep_word)
                continue
            current_brand_name = brand_recognizer(model_name, company_dict)['Company']

            #iterating every word
            for intent_name in word_dict:
              for item in word_dict[intent_name]:
                #searching the model from text
                if item[:3]==r'^(?':
                  a = re.findall(item , part,flags= re.IGNORECASE)

                else:
                  a = re.findall(r'\b'+item+r'\b',part,flags= re.IGNORECASE)
                
                
                model_name_found=re.search(item, model_name, flags= re.IGNORECASE)
                if a!=[] or a!=['']:
                  #if present
                  if model_name_found:
                    if '<insert>' in intent_name:
                      integer=[int(match.group()) for match in re.finditer(r'\d+', model_name)]
                      true_model_name = intent_name.replace('<insert>', str(integer[0]))

                    else:
                      true_model_name=intent_name

                  #findall gives a list of found words using the given regex so we loop through them
                  for variant in a:
                    if '<insert>' in intent_name:
                      integer=[int(match.group()) for match in re.finditer(r'\d+', variant)]
                      true_name = intent_name.replace('<insert>', str(integer[0]))
                    else:
                      true_name = intent_name
                    #if it is same as our model
                    if (model_name_found) and (true_name == true_model_name):
                      intents.append('Our Model')
                      last_intent = 'Our Model'

                    else:
                      if true_name==f'{current_brand_name}_General':
                        true_name='Parent_Company_General'
                        intents.append(true_name)
                        last_intent=true_name
                      else:
                        intents.append(true_name)
                        last_intent=true_name

                    matching_words.append(variant)
                    keywdsfnd.append(item)
            ##Special Case search Fold:
            #Check for Fold only without Iphone, google, pixel, samsung etc. and the model name is fold as well
            if re.search(r'(?<!iphone )(?<!pixel )(?<!samsung )(?<!google )(?<!galaxy )\bfold\b(?!(?:\s(?:iphone|pixel|samsung|google|galaxy)\b))', part, re.IGNORECASE) and not (true_model_name == 'Google Pixel Fold'):
               intents.append('Unspecified Brand Fold Phone')
               matching_words.append('fold')
               keywdsfnd.append(r'(?<!iphone )(?<!pixel )(?<!samsung )(?<!google )(?<!galaxy )\bfold\b(?!(?:\s(?:iphone|pixel|samsung|google|galaxy)\b))')
            
            intents=list(set(intents))

            if intents==[] and last_intent!='' and part!='[This review was collected as part of a promotion':
              mat=re.match(r"\bthis\b|\bit\b|\bit's\b", part, re.IGNORECASE)
              if mat and last_intent!='Our Model':
                intents = [last_intent]
                matching_words+=[mat.group()]
                current_col-=1
              else:
                last_intent=''
            #Adding the row
            df.loc[len(df)] = [og_df.iloc[current_row][ID_column], current_col, model_name, part, repr(sep_word), intents or ['Our Model'], matching_words,keywdsfnd]
            current_col += 1
        current_row += 1
        current_col = 1
    return df


def brand_recognizer(model, company_dict):
   """
   This function is used to recognize the brand from the model name.
   Args:
       model (str): Model name
       company_dict (dict): Dictionary containing the brand names and their regex
    Returns:
        pd.Series: Series containing the brand name
    """
   current_brand=None
   for i in company_dict:
     for j in company_dict[i]:
       if re.search(j, model, re.IGNORECASE):
         current_brand=i
         break
     if current_brand!=None:
       break
   return pd.Series({'Company':current_brand})

def merge_ambigious_sentences(group):
    """
    Utility function to merge sentences within each group
    Args:
        group (pd.DataFrame): Group of sentences
    Returns:
        pd.DataFrame: Merged sentences        
    """
    if len(group)>1:
      merged_sentence = ' '.join(group['Split'] + ' ' + group['Splitted_at'].apply(ast.literal_eval)).strip()
      # to_ret=pd.DataFrame({'UID': group['UID'].iloc[0],'Split_ID': group['Split_ID'].iloc[0], 'Split': merged_sentence, 'Splitted_at':group['Splitted_at'].iloc[-1], 'competitor_mentioned': group['competitor_mentioned'].iloc[0], 'competitor_word': group['competitor_word'].iloc[0], 'regex_word': group['regex_word'].iloc[0] })
      
      return pd.Series({'UID': group['UID'].iloc[0],'Split_ID': group['Split_ID'].iloc[0], 'Model Name':group['Model Name'].iloc[0], 'Split': merged_sentence, 'Splitted_at':group['Splitted_at'].iloc[-1], 'competitor_mentioned': group['competitor_mentioned'].iloc[0], 'competitor_word': group['competitor_word'].iloc[0], 'regex_word': group['regex_word'].iloc[0] })
    else:
      return pd.Series({'UID': group['UID'].iloc[0],'Split_ID': group['Split_ID'].iloc[0], 'Model Name':group['Model Name'].iloc[0], 'Split': group['Split'].iloc[0], 'Splitted_at':group['Splitted_at'].iloc[-1], 'competitor_mentioned': group['competitor_mentioned'].iloc[0], 'competitor_word': group['competitor_word'].iloc[0], 'regex_word': group['regex_word'].iloc[0] })


#Indirect Comparison
def indirect_comparison(file3_indirect, intent_dict, spw, func, base_path, onnx_path, tokenizer_path, intents, msgs=False):
    """
    This function is used to perform indirect comparison.
    Args:
        file3_indirect (pd.DataFrame): Dataframe containing the sentences where the comparison is indirect
        intent_dict (dict): Dictionary containing the keywords for each intent
        spw (list): List of special words
        func (function): Function to be used for sentiment analysis
        base_path (str): Base path of the project
    Returns:
        dict: Dictionary containing the sentiment for each intent
    """

    seperated_competetitors=sepearate_competitors(file3_indirect, intent_dict, spw, intents)

    for comp in seperated_competetitors:
        # dummy numbers
        for i in intents:
            
            if seperated_competetitors[comp][i].empty:
                if msgs:
                  print('Empty df for ')
                  print(comp, i)
                continue
            seperated_competetitors[comp][i].loc[:, i]=0

        for i in intents:
            if seperated_competetitors[comp][i].empty:
                    if msgs:
                      print('Empty df for ', i)
                    continue
            
            seperated_competetitors[comp][i].loc[:, i]=func(base_path, seperated_competetitors[comp][i], i, onnx_path=onnx_path, tokenizer_path=tokenizer_path )
            
        for i in intents:
            if seperated_competetitors[comp][i].empty:
                continue
            seperated_competetitors[comp][i][i]=seperated_competetitors[comp][i][i].apply(lambda x: x-1)
        if msgs:
          print(f'All Filling sentiment done for {comp}')

    seperated_competetitors=merge_comp_df(seperated_competetitors, msgs)

    return seperated_competetitors


def entities_sa(df, word_dict, spw, func, base_path, onnx_path, tokenizer_path, intents, msgs=False):
    """
    This function is used to perform sentiment analysis on specific cases where we have found primary, secondary or tertiary entity
    Primary: Enitity whose direct sentiment is to be found in the given text
    Secondary: Entity whose sentiment is reverse of the primary entity for the given review text
    Tertiary: Entity whose sentiment is 0 for the given review text
    Args:
        df (pd.DataFrame): Dataframe containing the sentences where the comparison is direct
        word_dict (dict): Dictionary containing the keywords for each intent
        spw (list): List of special words
        func (function): Function to be used for sentiment analysis
        base_path (str): Base path of Sentiment_Analysis directory where SA utils are 
    Returns:
        dict: Dictionary containing the sentiment for Model i.e. iphone 14, p7a etc. 
    """
    prim, sec, tert=split_df_to_prim_sec_tert(df)
    prim_seps=sepearate_competitors(prim, word_dict, spw, intents,  'Primary Entity')
    sec_seps=sepearate_competitors(sec, word_dict, spw, intents, 'Secondary_Entity')
    tert_seps=sepearate_competitors(tert, word_dict, spw , intents, 'Tertiary_Entity')


    for comp in prim_seps:
        # dummy numbers
        for i in intents:
            if prim_seps[comp][i].empty:
                if msgs:
                  print('Empty df for ')
                  print(comp, i)
                continue
                
            prim_seps[comp][i].loc[:, i]=0

        for i in intents:
            if prim_seps[comp][i].empty:
                    if msgs:
                      print('Empty df for ', i)
                    continue
            
            prim_seps[comp][i].loc[:, i]=func(base_path, prim_seps[comp][i], i, onnx_path=onnx_path, tokenizer_path=tokenizer_path )

        for i in intents:
            if prim_seps[comp][i].empty:
                continue
            prim_seps[comp][i][i]=prim_seps[comp][i][i].apply(lambda x: x-1)
        if msgs:
          print(f'All Filling sentiment done for {comp}')
    
    #Sentiment in secondary is just reverse of primary for same uid 
    for comp in sec_seps:
      for i in intents:
        if sec_seps[comp][i].empty:
            if msgs:
              print('Empty df for ')
              print(comp, i)
              continue
            vals=[]
            
            for id in sec_seps[comp][i].UID.to_list():
              tar_comp=sec[sec.UID==id]['Primary Entity'].values[0][0]
              vals += [prim_seps[tar_comp][i][prim_seps[tar_comp][i].UID==id][i].values[0]*-1]

            sec_seps[comp][i].loc[:, i]=vals
    
    #Sentiment in tertiary is just 0 regardless of primary and secondary
    for comp in tert_seps:
        for i in intents:
            if tert_seps[comp][i].empty:
                if msgs:
                  print('Empty df for ')
                  print(comp, i)
                continue
            tert_seps[comp][i].loc[:, i]=0
        # print('done')

    prim_fin=merge_comp_df(prim_seps, msgs)
    sec_fin=merge_comp_df(sec_seps, msgs)
    tert_fin=merge_comp_df(tert_seps, msgs)
    
    ##There is overlap in prim, sec and tert comps so we need to keep that in mind and merge for each comp at axis 0 i.e. rows
    final_res={}

    all_comps=list(prim_fin.keys())+list(sec_fin.keys())+list(tert_fin.keys()) 
    all_comps=list(set(all_comps))
    for comp in all_comps:
      result=concatenate_and_sort(prim_fin, sec_fin, tert_fin, competitor=comp, msgs=msgs)
      if result is not None:
          
        final_res[comp]=result

    return final_res

def transform_competitor_dictionary(fin_seperated_competetitors, intents):
    transformed={'UID':[] , 'Competitor_SA':[]}

    for comp in fin_seperated_competetitors:
      for row in fin_seperated_competetitors[comp].iterrows():
        row=row[1]
        temp_dict={}
        for i in intents:
          if i in row:
            temp_dict[i]=row[i]
          else:
            temp_dict[i]=np.NAN
        
        transformed['UID']+=[row['UID']]
        
        transformed['Competitor_SA']+=[{comp:temp_dict}]

    transformed_df=pd.DataFrame(transformed)
    return transformed_df


def merge_the_uids_competitor_dict(grp):
  """
  Utility function to create dictionary of the competitor sentiment analysis
  """
  if len(grp)>=2:
 
    merged_dict={}
    company_wise={}
    for dictionary in grp['Competitor_SA']:
      for k,v in dictionary.items():
        merged_dict[k]=v
        # brand=brand_recognizer(k, company_dict)
        # if brand['Company'].iloc[0] in company_wise:
        #   company_wise[brand['Company'].iloc[0]]+=[k]
        # else:
        #   company_wise[brand['Company'].iloc[0]]=[k]

    result={'UID':grp['UID'].to_list()[0],'Competitor_SA':merged_dict}

    return pd.Series(result)
  result={'UID':grp['UID'].to_list()[0],'Competitor_SA':grp['Competitor_SA'].to_list()[0]}
  return pd.Series(result)


def competitor_analyzer(base_path, base_intent_path, df, word_dict, spw, func, onnx_path, tokenizer_path, intents, Competitor_path = 'Intent_Dictionaries/Compare_dict.xlsx', msgs=True, dev_msgs=False):
    '''
    Main Competitor Analysis function which takes the dataframe and performs the competitor analysis on it.
    It divides the dataframe into 5 parts:
    1. Where only our model is mentioned
    2. Where only one competitor is mentioned
    3. Where more than one competitor is mentioned: 
      a. Where exactly 2 competitors are mentioned
      b. Where more than 2 competitors are mentioned
    then it resolves 2 amd 3a and performs sentiment analysis on them and then merges them.
    1 is returned back to pass through main pipeline
    Args:
        base_path (str): Path of the Sentiment_Analysis roberta folder
        base_intent_path (str): Path to the Intent analysis folder
        df (pd.DataFrame): Dataframe to be used for analysis
        word_dict (dict): Dictionary containing the keywords for each intent
        spw (dict): Dictionary with special words 
        func (function): Function to be used for sentiment analysis
        onnx_path (str): Path to the model
        tokenizer_path (str): Path to the tokenizer
        Competitor_path (str, optional): Path to the comparative dictionary. Defaults to 'Intent_Dictionaries/Compare_dict.xlsx'.
        msgs (bool, optional): Whether to print messages or not. Defaults to True.
        dev_msgs (bool, optional): Whether to print development messages or not. Defaults to False.
    '''
    if msgs:
        print('Competitor Analysis Ongoing...')
      
    model_dict, company_dict = load_comparative_dict(base_intent_path + Competitor_path)

    #Splitting the sentences and finding the competitor mentioned in the sentence
    if msgs:
        print('Splitting the sentences and finding the competitor mentioned in each sentence...')
    df = competitors_splitter(df, model_dict, company_dict)

    df = df.groupby(['UID', 'Split_ID'], group_keys=False).apply(merge_ambigious_sentences).reset_index(drop=True)
    
    if msgs:
        print('seperating the sentences according to competitors and given model...')
    #Where only and only our model is mentioned
    only_our_model = df[df['competitor_mentioned'].apply(lambda x: ['Our Model'] == x)] 
    
    #Where our model is not mentioned or other competitors are mentioned or both
    not_only_our_model=df[~(df['competitor_mentioned'].apply(lambda x: ['Our Model'] == x))] 
    

    #Merging the sentences where our model is mentioned, will be analyzed seperately
    temp_file = only_our_model.groupby('UID', group_keys=False).apply(merge_sentences).reset_index(drop=True) 

    #Where only one competitor is mentioned
    only_one_comp=not_only_our_model[not_only_our_model['competitor_mentioned'].apply(lambda x: len(x) == 1)]   

    #re Pattern for direct comparison
    combined_pattern = "|".join(Comparative_words)  

    #Where direct comparison is present
    one_comp_and_direct_comparison = only_one_comp[only_one_comp['Split'].str.contains(combined_pattern, regex=True)] 

    #Where direct comparison is not present
    one_comp_and_indirect_comparison = only_one_comp[~only_one_comp['Split'].str.contains(combined_pattern, regex=True)] 

    if msgs:
        print('Doing Indirect comp SA...')
    #Performing indirect comparison sentiment analysis
    if one_comp_and_indirect_comparison.empty:
      print('No indirect comp')
      indirect_comp={}
    else:
      indirect_comp=indirect_comparison(one_comp_and_indirect_comparison, word_dict, spw, func, base_path, onnx_path, tokenizer_path, intents=intents, msgs=dev_msgs) 

    #Where more than one competitor is mentioned
    more_than_one_comp = not_only_our_model[~(not_only_our_model['competitor_mentioned'].apply(lambda x: len(x) == 1))]  

    #Where exactly 2 competitors are mentioned
    two_comp=more_than_one_comp[(more_than_one_comp['competitor_mentioned'].apply(lambda x: len(x) == 2))]   
    
    #Where more than 2 competitors are mentioned
    more_than_two_comp=more_than_one_comp[~(more_than_one_comp['competitor_mentioned'].apply(lambda x: len(x) == 2))]  

    if msgs:
       print('Resolving the sentences...')
    #Where more than 2 competitors are mentioned and direct comparison is present
    than_file=pd.concat([one_comp_and_direct_comparison[one_comp_and_direct_comparison['Split'].str.contains(r'|'.join(\
       words_where_senti_for_first_and_second_is_reverse), regex=True)],\
       two_comp[two_comp['Split'].str.contains(r'|'.join(words_where_senti_for_first_and_second_is_reverse)\
                                         , regex=True)]], axis=0)  #File 3 direct and file 5 combined where the
                                                                   #sentiment for first and second is reverse i.e.
                                                                   #worse than, better than etc.
    
    than_file.sort_values('UID', inplace=True)
    than_file.reset_index(drop=True, inplace=True)

    
    remaining_df=pd.concat([one_comp_and_direct_comparison[~one_comp_and_direct_comparison['Split'].str.contains\
                                         (r'|'.join(words_where_senti_for_first_and_second_is_reverse), regex=True)],\
                             two_comp[~two_comp['Split'].str.contains\
                                   (r'|'.join(words_where_senti_for_first_and_second_is_reverse), regex=True)]], axis=0)

    
    equivalent_file=remaining_df[remaining_df['Split'].str.contains(r'|'.join(words_where_senti_remains_same_for_both), regex=True, case=False)]
    remaining_df=remaining_df[~(remaining_df['Split'].str.contains(r'|'.join(words_where_senti_remains_same_for_both), regex=True, case=False))]
    
    transition_file=remaining_df[remaining_df['Split'].str.contains(r'|'.join(eternal_our_model_comp_words), regex=True, case=False)]
    remaining_df=remaining_df[~(remaining_df['Split'].str.contains(r'|'.join(eternal_our_model_comp_words), regex=True, case=False))]

    from_file=remaining_df[remaining_df['Split'].str.contains(r'|'.join(from_word), regex=True, case=False)]
    remaining_df=remaining_df[~(remaining_df['Split'].str.contains(r'|'.join(from_word), regex=True, case=False))]

    as_file=remaining_df[remaining_df['Split'].str.contains(r'\bas\b', regex=True, case=False)]
    remaining_df=remaining_df[~(remaining_df['Split'].str.contains(r'\bas\b', regex=True, case=False))]

    #Add more than 2 competitors to remaining df
    remaining_df = pd.concat([remaining_df, more_than_two_comp], axis=0).sort_values('UID').reset_index(drop=True)
    

    # resolving above files
    equi_file_resolved=equivalent_file.apply(equivalent_resolver, axis=1)

    transition_file_resolved=transition_file.apply(transition_resolver, axis=1)

    from_file_resolved=from_file.apply(lambda row: from_resolver(row, model_dict, msgs=dev_msgs), axis=1)

    as_file_resolved=as_file.apply(lambda row: as_resolver(row, model_dict, msgs=dev_msgs), axis=1)

    than_file_resolved=than_file.apply(lambda row: entity_in_direct_comp_words_identifier(row, model_dict, msgs=dev_msgs), axis=1)
    
    if msgs:
      print('Ongoing SA on resolved files...')

    #SA on all resolved files
    if msgs:
      print('Ongoing SA on reverse resolved files...')
    if than_file_resolved.empty:
      print('No than file')
      than_SA_file={}
    else:
      than_SA_file=entities_sa(than_file_resolved, word_dict, spw, func, base_path, onnx_path, tokenizer_path, intents=intents , msgs=dev_msgs)
    
    if msgs:
      print('Ongoing SA on equal resolved files...')
    if equi_file_resolved.empty:
      print('No equi file')
      equi_SA_file={}
    else:
      equi_SA_file=entities_sa(equi_file_resolved, word_dict, spw, func, base_path, onnx_path, tokenizer_path, intents=intents, msgs=dev_msgs)
    if msgs:
      print('Ongoing SA on transition resolved files...')
    if transition_file_resolved.empty:
      print('No transition file')
      transition_SA_file={}
    else:
      transition_SA_file=entities_sa(transition_file_resolved, word_dict, spw, func, base_path, onnx_path, tokenizer_path, intents=intents, msgs=dev_msgs)
    if msgs:
      print('Ongoing SA on from resolved files...')
    if from_file_resolved.empty:
      print('No from file')
      from_SA_file={}
    else:
      from_SA_file=entities_sa(from_file_resolved, word_dict, spw, func, base_path, onnx_path, tokenizer_path, intents=intents, msgs=dev_msgs)
    if msgs:
      print("Ongoing SA on special case 'as' resolved files...")
    if as_file_resolved.empty:
      print('No as file')
      as_SA_file={}
    else: 
      as_SA_file=entities_sa(as_file_resolved, word_dict, spw, func, base_path, onnx_path, tokenizer_path, intents=intents, msgs=dev_msgs)
    
    #Merging all the files
    if msgs:
      print('Merging all the files...')
    all_comp_files=list(indirect_comp.keys())+list(than_SA_file.keys())+list(equi_SA_file.keys())+list(transition_SA_file.keys())+list(from_SA_file.keys())+list(as_SA_file.keys())
    all_comp_files=list(set(all_comp_files))
    final_comp_dfs={}

    for comp in all_comp_files:
      result=concatenate_and_sort(indirect_comp, than_SA_file, equi_SA_file, transition_SA_file, from_SA_file, as_SA_file, competitor=comp, msgs=dev_msgs)
      if result is not None:
        # result=result.groupby('UID', group_keys=False).apply(merge_sentences).reset_index(drop=True)
        final_comp_dfs[comp]=result
    if msgs:
      print('Merging done')
    
    ##Tranforming the dataframes
    final_comp_dfs=transform_competitor_dictionary(final_comp_dfs, intents)
    final_comp_dfs=final_comp_dfs.groupby('UID', group_keys=False).apply(merge_the_uids_competitor_dict).reset_index(drop=True)
    
    
    Our_model_comb=concatenate_and_sort(than_SA_file, equi_SA_file, transition_SA_file, from_SA_file, as_SA_file, competitor='Our Model', msgs=dev_msgs)
    
    parent_company_comb=concatenate_and_sort(indirect_comp, than_SA_file, equi_SA_file, transition_SA_file, from_SA_file, as_SA_file, competitor='Parent_Company_General', msgs=dev_msgs)


    if msgs:
      print('Competitor Analysis Done\n')
    # indirect_comp, equi_file_resolved, transition_file_resolved, from_file_resolved, as_file_resolved, than_file_resolved

    return temp_file, final_comp_dfs, remaining_df, Our_model_comb, parent_company_comb