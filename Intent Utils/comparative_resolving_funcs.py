import re
import pandas as pd
from comprative_config import words_where_senti_for_first_and_second_is_reverse, from_word
# from config import current_possible_intents as Mobile_intents
import numpy as np
from intent_utils import create_final_dataframe, merge_sentences, join_at_stopwords


############################ General Func and Utility Functions #############################################
def find_entity(sentence, word_dict, model_name):
  entities={}
  for intent_name in word_dict:
    for item in word_dict[intent_name]:
      # searching the model from text
      if item[:3]==r'^(?':
        a = re.findall(item , sentence, re.IGNORECASE)
        model_name_found=re.match(item, model_name, re.IGNORECASE)
      else:
        a = re.findall(r'\b'+item+r'\b', sentence, re.IGNORECASE)
        model_name_found=re.match(item, model_name, re.IGNORECASE)
      if a!=[] or a!=['']:
        #if present
        if model_name_found:
          if '<insert>' in intent_name:
            integer=[int(match.group()) for match in re.finditer(r'\d+', model_name)]
            true_model_name = intent_name.replace('<insert>', str(integer[0]))
          else:
            true_model_name=model_name

        #findall gives a list of found words using the given regex so we loop through them
        for variant in a:
          if '<insert>' in intent_name:
            integer=[int(match.group()) for match in re.finditer(r'\d+', variant)]
            true_name = intent_name.replace('<insert>', str(integer[0]))
          else:
            true_name = intent_name
          #if it is same as our model
          if (model_name_found) and (true_name == true_model_name):
            if 'Our Model' in entities:
              entities['Our Model']+=[variant]
            else:
              entities['Our Model']=[variant]
            # entities.append(f'{true_name}<<sep>>{variant}<<sep>>Our Model')
          else:
            if  true_name in entities:
              entities[true_name]+=[variant]
            else:
              entities[true_name]=[variant]
            # entities.append(f'{true_name}<<sep>>{variant}')

  return entities

def merge_comp_df(df, msgs=False):
  df2={}
  for comp in df:
    overall_df = pd.concat(df[comp].values(), axis=0)
    
    if overall_df.empty or len(overall_df.columns)<=2:
        if msgs:
          print('No significant sentiment found for ', comp)
    else:
        overall_df.sort_values('UID', inplace=True)
        overall_df.drop('Review Text', axis=1, inplace=True)
        merged_df = overall_df.groupby('UID').agg(lambda x: x.dropna().iloc[0] if x.notna().any() else np.nan).reset_index()
        df2[comp]=merged_df
  return df2


def split_df_to_prim_sec_tert(df):
    primary_df=df[df['Primary Entity'].apply(lambda x: x!=[])]
    secondary_df=df[df['Secondary_Entity'].apply(lambda x: x!=[])]
    tertiary_df=df[df['Tertiary_Entity'].apply(lambda x: x!=[])]
    return primary_df, secondary_df, tertiary_df

def sepearate_competitors(df, word_dict, spw, intents,  comp_col='competitor_mentioned') :
    seperated_competetitors={}
    
    for competetitors in df[comp_col]:
        for competetitor in competetitors:
          if competetitor not in seperated_competetitors:
            seperated_competetitors[competetitor]=df[df[comp_col].apply(lambda x: competetitor in x)]

            merg_df = seperated_competetitors[competetitor].groupby('UID', group_keys=False).apply(merge_sentences).reset_index(drop=True)
            temp_df=create_final_dataframe(merg_df, word_dict, spw).reset_index(drop=True)
            temp_df = join_at_stopwords(temp_df)
            seps={}
            for i in intents:
                filtered_df = temp_df[temp_df['Intent'].apply(lambda x: i in x)]
                ids=filtered_df.UID.to_list()
                seps[i] = merg_df[merg_df.UID.isin(ids)]
            # seps['General']=merg_df
            seperated_competetitors[competetitor]=seps
    return seperated_competetitors

def apply_conditions(x):
  # print(x)
    if x.isna().all():
      return np.nan
    total_sum = x.dropna().sum()
    if pd.notna(total_sum): 
        if total_sum >= 1:
            return 1
        elif total_sum <= -1:
            return -1
        else:
            return 0
    else:
        return np.nan  
    
def concatenate_and_sort(*dfs, competitor, msgs=False):
      competitor_dfs = [df[competitor] for df in dfs if competitor in df]
      
      if competitor_dfs:
          result = pd.concat(competitor_dfs, axis=0).sort_values('UID').reset_index(drop=True)
          result = result.groupby('UID', group_keys=False).agg(lambda x: apply_conditions(x)).reset_index()

          return result
      else:
          if msgs:
              print(f'Competitor data not present for {competitor}')
          return None
      

#########################################Resolver Funcs for each case#############################################
##Than Resolver
def entity_in_direct_comp_words_identifier(row, model_dict, model_col_name='Model Name', text_col_name='Split', msgs=False):
  sentence=row[text_col_name]
  m_name=row[model_col_name]

  #Assumption Only one than present at a time in a sentence so we get 2 splits each time
  splits = re.split(r'|'.join(words_where_senti_for_first_and_second_is_reverse), sentence, flags=re.IGNORECASE, maxsplit=1)
  entities1=find_entity(splits[0], model_dict, m_name)
  if entities1=={}:
    primary_entities=['Our Model']
  else:
    primary_entities=list(entities1.keys())

  entities2=find_entity(splits[1], model_dict, m_name)
  if entities2=={}:
    secondary_entities=['Our Model']
  else:
    secondary_entities=list(entities2.keys())
  if secondary_entities==primary_entities:
    if msgs:
      print('Weird Comparision for UID', row['UID'])
      print('Sentence')
      print(row['Split'])
      print()

    secondary_entities=[]
    

  #Cases where primary n secondary entity
  for i in primary_entities:
    if i in secondary_entities:
      secondary_entities.remove(i)


  tertiary_entities=[]

  result={i:j for i, j in row.to_dict().items()}
  result['Primary Entity']=primary_entities
  result['Secondary_Entity']=secondary_entities
  result['Tertiary_Entity']=tertiary_entities

  return pd.Series(result)


##Equivalent
def equivalent_resolver(row, model_col_name='Model Name', text_col_name='Split'):
  sentence=row[text_col_name]
  m_name=row[model_col_name]
  primary_entities=row['competitor_mentioned']
  secondary_entities=[]
  tertiary_entities=[]
  if len(primary_entities)==1 and not 'Our Model' in primary_entities:
    primary_entities+=['Our Model']

  result={i:j for i, j in row.to_dict().items()}
  result['Primary Entity']=primary_entities
  result['Secondary_Entity']=secondary_entities
  result['Tertiary_Entity']=tertiary_entities

  return pd.Series(result)


#Transitioned from some phone to some 
def transition_resolver(row, model_col_name='Model Name', text_col_name='Split'):
  sentence=row[text_col_name]
  m_name=row[model_col_name]

  primary_entities=['Our Model']
  secondary_entities=[]
  tertiary_entities=row['competitor_mentioned']
  if 'Our Model' in row['competitor_mentioned']:
    tertiary_entities.remove('Our Model')

  result={i:j for i, j in row.to_dict().items()}
  result['Primary Entity']=primary_entities
  result['Secondary_Entity']=secondary_entities
  result['Tertiary_Entity']=tertiary_entities

  return pd.Series(result)

def from_resolver(row, model_dict, model_col_name='Model Name', text_col_name='Split', msgs=False):
  sentence=row[text_col_name]
  m_name=row[model_col_name]

  #Assumption Only one than present at a time in a sentence so we get 2 splits each time
  splits = re.split(r'|'.join(from_word), sentence, flags=re.IGNORECASE, maxsplit=1)
  entities1=find_entity(splits[0], model_dict, m_name)

  if entities1=={}:
    primary_entities=['Our Model']
  else:
    primary_entities=list(entities1.keys())

  entities2=find_entity(splits[1], model_dict, m_name)
  if entities2=={}:
    tertiary_entities=['Our Model']
  else:
    tertiary_entities=list(entities2.keys())
  if tertiary_entities==primary_entities:
    if msgs:
      print('Weird Comparision for UID', row['UID'])
      print('Sentence')
      print(row['Split'])
      print()

    tertiary_entities=[]

  #Cases where primary n secondary entity
  for i in primary_entities:
    if i in tertiary_entities:
      tertiary_entities.remove(i)
  
  secondary_entities=[]

  result={i:j for i, j in row.to_dict().items()}
  result['Primary Entity']=primary_entities
  result['Secondary_Entity']=secondary_entities
  result['Tertiary_Entity']=tertiary_entities

  return pd.Series(result)

def as_resolver(row, model_dict, model_col_name='Model Name', text_col_name='Split', msgs=False):
  sentence=row[text_col_name]
  m_name=row[model_col_name]
  
  not_pattern=r'\bnot\s*as\b'
  if re.match(not_pattern, sentence, flags=re.IGNORECASE):  
    #Assumption Only one than present at a time in a sentence so we get 2 splits each time
    splits = re.split(not_pattern, sentence, flags=re.IGNORECASE, maxsplit=1)

    entities1=find_entity(splits[0], model_dict, m_name)

    if entities1=={}:
      primary_entities=['Our Model']
    else:
      primary_entities=list(entities1.keys())

    entities2=find_entity(splits[1], model_dict, m_name)
    if entities2=={}:
      secondary_entities=['Our Model']
    else:
      secondary_entities=list(entities2.keys())
    if secondary_entities==primary_entities:
      if msgs:
        print('Weird Comparision for UID', row['UID'])
        print('Sentence')
        print(row['Split'])
        print()

      secondary_entities=[]
  
    #Cases where primary n secondary entity overlap
    for i in primary_entities:
      if i in secondary_entities:
        secondary_entities.remove(i)

  else:
      primary_entities=row['competitor_mentioned']
      secondary_entities=[]
  
  result={i:j for i, j in row.to_dict().items()}
  result['Primary Entity']=primary_entities
  result['Secondary_Entity']=secondary_entities
  result['Tertiary_Entity']=[]

  return pd.Series(result)
