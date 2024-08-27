from intent_utils import *
from finetune_roberta import *
import copy 
from transformers import AutoTokenizer, RobertaForSequenceClassification
# from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
from pro_con_sa import Pro_Con_Handler
from dataframe_preprocessing import *
# from keywds_analysis import *
# from config import current_possible_intents as Mobile_intents
import zipfile
import os
from competitor_analysis import competitor_analyzer

############################################################
def SA_fetch(model, dataset, device):
  """
    SA for given model and dataset, returns the sentiment for each text or rather the batches of text
  """
  # Evaluation settings (Not needed for ORT models)
  # model.eval()
  res=[]
  # Iterate over the dataset
  for batch in dataset:
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      labels = batch['label'].to(device)

      # Disable gradient calculations
      with torch.no_grad():
          # Forward pass
          outputs = model(input_ids, attention_mask=attention_mask)
          logits = outputs.logits

      # Predicted labels
      batch_predicted_labels = torch.argmax(logits, dim=1)
      res+=batch_predicted_labels.cpu().detach().tolist()
  return res

def func(base_path, df_to_use, column_name, Text_column_name='Review Text', onnx_path=None, tokenizer_path=None):
    """
    This func does the SA for the given dataframe of a certain domain/intent, It loads the model and tokenizer,
      readies the dataset and calls the SA_fetch function which gives the setiment for each text or rather the batches of text
    Args:
        base_path (str): Path to the folder containing the models and tokenizers, and Sentiment Utils
        df_to_use (pd.DataFrame): Dataframe to be used for analysis
        column_name (str): Name of the column where the sentiment is to be filled/ Intent column name
        Text_column_name (str, optional): Name of the column where the text is present. Defaults to 'Review Text'.
        onnx_path (str, optional): Path to the onnx model. Defaults to latest working model.
        tokenizer_path (str, optional): Path to the tokenizer. Defaults to latest working model.
    """
    
    df=df_to_use[[Text_column_name,column_name]]

    if onnx_path is None:
        # onnx_path=base_path+f"seperated_try_Models/{column_name}/model_20_oct/model-optimized"
        # onnx_path=base_path+f"seperated_try_Models/{column_name}/model_1_Nov-optimized"
        # onnx_path=f'/content/drive/Shareddrives/SA/{column_name}/Model_DeBERTa_V3_14_Dec'
        onnx_path=f'/content/drive/Shareddrives/SA/{column_name}/Model_DeBERTa_V3_7_Jan'
    else:
        if bool(re.search(r'\{\}', onnx_path)):
            onnx_path=onnx_path.format(column_name)
        else:
            onnx_path=onnx_path

    
    # model=ORTModelForSequenceClassification.from_pretrained(onnx_path, file_name="model_optimized.onnx", use_io_binding=True)
    if 'DeBERTa' in onnx_path:
        model=DebertaV2ForSequenceClassification.from_pretrained(onnx_path)
    else:
        # model=ORTModelForSequenceClassification.from_pretrained(onnx_path, file_name="model_optimized.onnx", use_io_binding=True)
        model=RobertaForSequenceClassification.from_pretrained(onnx_path)
        
    if tokenizer_path is None:    
        # tokenizer_path=base_path+f'seperated_try_Models/{column_name}/tokenizer_20_oct'
        # tokenizer_path=f'/content/drive/Shareddrives/SA/{column_name}/scratch_tokenizer_31_oct'
        tokenizer_path = f'/content/drive/Shareddrives/SA/{column_name}/Tokenizer_V3_DeBERTa_7_Jan'
    else:
        if bool(re.search(r'\{\}', tokenizer_path)):
            tokenizer_path=tokenizer_path.format(column_name)
        else:
            tokenizer_path=tokenizer_path

    # tokenizer = AutoTokenizer.from_pretrained(base_path+f'seperated_try_Models/{column_name}/tokenizer_20_oct')
    # tokenizer = AutoTokenizer.from_pretrained(f'/content/drive/Shareddrives/SA/{column_name}/scratch_tokenizer_31_oct')
    if 'DeBERTa' in tokenizer_path:
        tokenizer = DebertaV2Tokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path, is_split_into_words=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df.loc[:,Text_column_name] = df[Text_column_name].apply(preprocess_text)

    dataset = SentimentDataset(df, tokenizer, Text_column_name, column_name)
    changed_order = dataset.changed_order
    dataloader = DataLoader(dataset, batch_size=128, collate_fn=lambda batch: dynamic_collate_fn(batch, tokenizer))
    model.to(device)
    lis=SA_fetch(model, dataloader, device)
    # Create a mapping from the given indices to the natural order
    index_mapping = {index: i for i, index in enumerate(changed_order)}
    fin_lis = [None] * len(lis) 
    for original_index, new_index in index_mapping.items():
        fin_lis[original_index] = lis[new_index]
   

    del model
    return fin_lis



def supply_df(base_path, base_intent_path, og_df, intents, msgs=True, dev_msgs=False, onnx_path=None, tokenizer_path=None, Intent_dict_path_and_sheet_info=['Intent_Dictionaries/Mobile_Intent_dict.xlsx', 'MOBILE', 'Special_words_Mobile'], Competitor_path='Intent_Dictionaries/Compare_dict.xlsx'):
    """
    This function is used to supply the dataframe to the SA module.
    It loads the intent dictionary and stopwords and then calls the competitor_analyzer function.
    It then calls the Pro_Con_Handler function if there are any pros and cons present.
    It then calls the create_final_dataframe function to create the final dataframe.
    It then calls the join_at_stopwords function to join the sentences at stopwords.
    then does the SA for each domain/intent and fills the sentiment.
    Finally it merges all output dataframes and returns the final dataframe.
    Args:
        base_path (str): Path to the folder containing the models and tokenizers, and Sentiment Utils
        base_intent_path (str): Path to the folder containing the intent dictionaries and stopwords
        og_df (pd.DataFrame): Original dataframe supplied by the user
        msgs (bool, optional): Whether to print messages or not. Defaults to True. like what is ongoing
        dev_msgs (bool, optional): Whether to print development messages or not. Defaults to False. like any errors or warnings
        onnx_path (str, optional): Path to the onnx model. Defaults to None i.e. latest working model.
        tokenizer_path (str, optional): Path to the tokenizer. Defaults to None i.e. latest working model.
        Intent_dict_path (str, optional): Path to the intent dictionary from Intent_analysis folder. Defaults to 'Intent_Dictionaries/Mobile_Intent_dict.xlsx'.
        Competitor_path (str, optional): Path to the competitor dictionary from Intent_analysis folder. Defaults to 'Intent_Dictionaries/Compare_dict.xlsx'.
        intents (list, optional): List of intents for which SA is possible rn. Defaults to intents of Mobile.
    Returns:
        pd.DataFrame: Final dataframe with sentiment filled
    """

    df=copy.deepcopy(og_df)
    intent_dict, spw = load_intent_dict(base_intent_path+Intent_dict_path_and_sheet_info[0], Intent_dict_path_and_sheet_info[1], Intent_dict_path_and_sheet_info[2])
    
    #Preprocessing
    df = process_df(df)

    df, final_comp_df, remaining_df, our_model_temp, parent_comp_temp = competitor_analyzer(base_path, base_intent_path, df, intent_dict, spw, func, onnx_path, tokenizer_path, Competitor_path=Competitor_path, intents=intents, msgs=msgs,dev_msgs=dev_msgs)

    #Pros and Cons
    if msgs:
        print('Seperating Pros and Cons...')
    pattern1 = r'\bpros\b\s?:|\bCons\b\s?:'
    pattern2 = r'\'\s?.\s?\''
    cond=df['Review Text'].str.contains(pattern1, case=False, regex=True) & df['Review Text'].str.contains(pattern2, case=False, regex=True)
    
    #Seperate Pros and Cons and remove from main df 
    pros_cons = df[cond]
    df = df[~cond]
    if not pros_cons.empty:
        #Handling Pros and Cons
        if msgs:
            print('Handling Pros and Cons...')

        procondf=Pro_Con_Handler(base_path, pros_cons, intent_dict, spw, func, intents, msgs, onnx_path, tokenizer_path)
    else:
        procondf=None
        if msgs:
            print('Warning: No Pros and Cons found', end='\n\n')
    if msgs:
        print('Going on with regular reviews...')
        print('Splitting sentences and assigning intent...')
    
    watch_dict2={}
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
        watch_dict2[i]=temp

    final_df = create_final_dataframe(df, watch_dict2, spw)
    if msgs:
        print('Joining at stopwords(except: sentiment reversers like however, but, although etc.)...')

    final_df = join_at_stopwords(final_df)
    
    if msgs:
        print('done')
    # if choice in [1,3]:
    #     print('Keyword Analysis...')
    #     key_fin_dfs=keyword_analyzer(base_path, final_df, intent_dict, spw, intents, func)
    # if choice in [2,3]:
        # print('Sentiment Analysis...')
    if msgs:
        print('Seperating by domain/intent...')
    seperated_dfs=merge_like_sentences_without_senti(final_df, intents)
    
    if msgs:
        print('Seperating by domain/intent done')
    
    print('Filling sentiment...')
    ##Sentiment Filling

    # dummy numbers
    for i in intents:
        # seperated_dfs[i]=pd.read_excel(f"{i}_file.xlsx")
        seperated_dfs[i][i]=0

    for i in intents:
        if msgs:
            print(f'filling sentiment for {i}...')

        if seperated_dfs[i].empty:
            if msgs:
                print('Empty df for ', i)
            continue
        seperated_dfs[i].loc[:, i]=func(base_path,df_to_use = seperated_dfs[i], column_name = i , onnx_path = onnx_path, tokenizer_path = tokenizer_path)
        if msgs:
            print('done')        
    for i in intents:
        if seperated_dfs[i].empty:
            continue
        seperated_dfs[i][i]=seperated_dfs[i][i].apply(lambda x: x-1)
    if msgs:
        print('All Filling sentiment done')

    overall_df = pd.concat(seperated_dfs.values(), axis=0)
    if overall_df.empty:
        if msgs:
            print('No significant sentiment found')
        return 
    
    overall_df.sort_values('UID', inplace=True)
    overall_df.drop('Review Text', axis=1, inplace=True)
    merged_df = overall_df.groupby('UID').agg(lambda x: x.dropna().iloc[0] if x.notna().any() else np.nan).reset_index()

    if procondf is None:
        combined_df = merged_df
    else:
        combined_df = pd.concat([merged_df, procondf], axis=0).sort_values('UID').reset_index(drop=True)
    
    columns_to_keep=['Name', 'UID', 'Model Name', 'Model Number', 'Country', 'Brand', 'Source', 'Rating', 'Review Date', "Seeded Review", 'Review Title', 'Review Text', 'Native_Review_Title', 'Native_Review_Text']

    columns_present=og_df.columns.to_list()
    columns_to_drop=list(set(columns_present)-set(columns_to_keep))
    df2=og_df.drop(columns_to_drop, axis=1)
    
    # column_to_drop_after = 'Review Text'  
    # column_index = og_df.columns.get_loc(column_to_drop_after) + 1

    # df2 = og_df.iloc[:, :column_index]

    # resdf = pd.concat([df2.set_index('UID'), combined_df.set_index('UID')], axis=1).reset_index()
    resdf = pd.merge(df2, combined_df, how='left', on='UID')
    
    # resdf = pd.concat([resdf.set_index('UID'), final_comp_df.set_index('UID')], axis=1).sort_values('UID').reset_index()
    resdf = pd.merge(resdf, final_comp_df, how='left', on='UID')


    ###If intent is null in resdf but present in temp_model_combined_df then fill it
    if isinstance(our_model_temp, pd.DataFrame):
        for i in intents:
            if i in our_model_temp.columns:
                cond=resdf.UID.isin(our_model_temp[our_model_temp[i].notna()].UID.to_list())
            else:
                continue
            if pd.isnull(resdf.loc[cond, i]).any():
                #Fill the null values with the values from temp_model_combined_df
                cond2=resdf.loc[cond, i].isnull()
                final_cond=cond & cond2
                resdf.loc[final_cond, i]=our_model_temp.loc[our_model_temp.UID.isin(resdf[final_cond].UID.to_list()), i].values

    ###If intent is null in resdf but present in parent_comp_temp then fill it
    if isinstance(parent_comp_temp, pd.DataFrame):
        for i in intents:
            if i in parent_comp_temp.columns:
                cond=resdf.UID.isin(parent_comp_temp[parent_comp_temp[i].notna()].UID.to_list())
            else:
                continue
            if pd.isnull(resdf.loc[cond, i]).any():
                #Fill the null values with the values from parent_comp_temp
                cond2=resdf.loc[cond, i].isnull()
                final_cond=cond & cond2
                resdf.loc[final_cond, i]=parent_comp_temp.loc[parent_comp_temp.UID.isin(resdf[final_cond].UID.to_list()), i].values

    
    return resdf, remaining_df
    # return {'sentiment':resdf, 'keyword':key_fin_dfs}
