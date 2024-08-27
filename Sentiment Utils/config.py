
######################Latest Model Paths####################
MOBILE_MODEL_PTH = '/content/drive/Shareddrives/SA/Mobile_Models/{}/scratch_model_21_Jan'
MOBILE_TOKENIZER_PTH = '/content/drive/Shareddrives/SA/Mobile_Models/{}/scratch_tokenizer_21_Jan'

WATCH_MODEL_PTH = '/content/drive/Shareddrives/SA/Watch_Models/{}/scratch_model_12_Apr'
WATCH_TOKENIZER_PTH = '/content/drive/Shareddrives/SA/Watch_Models/{}/scratch_tokenizer_12_Apr'


######################Intent Dictionary Paths################
Mobile_intent_dict_pth_and_sheet_info = ['Intent_Dictionaries/Mobile_Intent_dict.xlsx', 'MOBILE', 'Special_words_Mobile']
Watch_intent_dict_pth_and_sheet_info =['Intent_Dictionaries/Watch_Intent_dict.xlsx','Watches', 'Special_words_Watch']
Buds_intent_dict_pth_and_sheet_info = ['Intent_Dictionaries/buds_ANC_Intent_dict.xlsx', 'buds', 'Special_words_Buds']
Tablet_intent_dict_pth_and_sheet_info = ['Intent_Dictionaries/Tablet_Intent_dict.xlsx', 'Tablet', 'Special_words_Tablet']


######################Competitor Dictionary Paths################
Mobile_competitor_dict_pth = 'Intent_Dictionaries/Compare_dict.xlsx'
Watch_competitor_dict_pth = 'Intent_Dictionaries/Compare_dict_Watch.xlsx'

#########################Intent Possible Lists##################
#Intents for which SA is possible rn
current_possible_intents=['Price', 'Build and Design', 'Screen', 'Camera',	'Battery'\
                          ,	'Performance_Processor',	'Software',	'NETWORKING',	'Other_Specs']

current_possible_watch_intents=['Price',
 'Screen',
 'Software_OS_Performance',
 'Sensors_Tracker',
 'Battery',
 'Connectivity_Setup',
 'Build and Design',
]

current_possible_tablet_intents=['Price', 'Build and Design', 'Screen', 'Camera',	'Battery'\
                          ,	'Performance_Processor',	'Software',	'NETWORKING',	'Other_Specs', 'Docker']


######################Intent Splitting Keywords################
#Specifies Intent Splitting keywds
sep = ['\n', '.', ',','?', '!','<<TB4>>',';']
stpwds = ['and', 'but', 'or', 'although', 'however','nd', 'althgh', 'hwvr']

#To keep joined after splitting and intent analysis
to_keep_joined=['\',\'', '\'and\'', '\'or\'', '\'!\'', '\';\'']


###########################Stopwords############################
#List of stopwords to be removed from the text before sentiment analysis, used in finetune_roberta script

Stopwords_list= [
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
    'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should',
    "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
    'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
]


######################Human Error Correction##################
#Conversion dictionary to map human error to correct intent label
mobile_human_error_dict={'cam':'Camera',\
                'Batt':'Battery',\
                'build':'Build and Design',\
                'Screen|Display': 'Screen',\
                'Other|specs': 'Other_Specs',\
                'gam':'Gaming',\
                'Custom': 'Customer Service',\
                'Speak':'Speaker',\
                'stor':'Storage',\
                r'perform|\bpro|processor':'Performance_Processor',\
                'soft':'Software',\
                'Net':'NETWORKING',\
                'Price':'Price',\
                'Deliv':'Delivery',\
                r'Unclassified|general|\bge|comparitive experience': 'Unclassified',\
                'Access':'Accessories'}

#Conversion dictionary to map human error to correct intent label
watch_human_error_dict={
                # 'cam':'Camera',\
                'Batt':'Battery',\
                'build':'Build and Design',\
                'Screen|Display': 'Screen',\
                'Sensors?|Tracker': 'Sensors_Tracker',\
                'Custom': 'Customer Service',\
                'Connectivity|Setup':'Connectivity_Setup',\
                'stor':'Storage',\
                'Misc|other':'Misc Features',\
                r'soft|performance|processor|\bos\b':'Software_OS_Performance',\
                'Subscription':'Subscription',\
                'Price':'Price',\
                'Deliv':'Delivery',\
                r'Unclassified|general|\bge|comparitive experience': 'Unclassified',\
                'Access':'Accessories'}

buds_human_error_dict={
                'Batt':'Battery',\
                'build':'Build and Design',\
                'Touch|control|gestures?|Sensors?': 'Touch Control_Gestures',\
                'sound': 'Sound_Quality',\
                'Custom': 'Customer Service',\
                'Connectivity|Setup':'Connectivity_Setup',\
                'Misc|other':'Misc Features',\
                r'Soft|\bos\b':'Software_OS',\
                r'\bEcosystem':'Ecosystem',\
                r'\bai\b|assist|smart':'AI_Assistance',\
                'noise|cancel':'Automatic_Noise_Cancellation',\
                'Transparency': 'Transparency',\
                r'performance|processor|\bos\b':'Performance_Processor',\
                r'\bmic\b|microphone':'Microphone',\
                'Price':'Price',\
                'Deliv':'Delivery',\
                r'Unclassified|general|\bge|comparitive experience': 'Unclassified',\
                'Access':'Accessories'}


tablet_human_error_dict={'[cC]am|camera':'Camera',\
                'Batt|battery':'Battery',\
                '[bB]uild(?:\s&\s)?\/?[dD]esign|[Bb]uild':'Build and Design',\
                '[sS]creen(?:\s&\s)?\/?[Dd]isplay|Screen|Display': 'Screen',\
                'Other|specs': 'Other_Specs',\
                'gam':'Gaming',\
                'Custom': 'Customer Service',\
                'Speak':'Speaker',\
                'stor':'Storage',\
                r'[Pp]erform|[Pp]erformance|[pP]erforma|\b[Pp]ro|[pP]rocessor':'Performance_Processor',\
                '[sS]oftware\/?\s?&?\s?Setup|s\[sS]oft':'Software',\
                '[eE]co':'EcosystemEcosystem',\
                r'[cC]onnectivity(?:\([WwiFfi]+\+[cC]ellular\))|connect':'NETWORKING',\
                '[oO]ther\s[Ss]pecs':'Other_Specs',\
                'Deliv':'Delivery',\
                'Access|accessories':'Accessories',\
                'speaker':'Speaker',\
                'gaming':'Gaming',\
                '[pP]rice':'Price',\
                r'Unclassified|general|\bge|comparitive experience': 'Unclassified',\
                'Dock': 'Docker' ,\
                r'\bEcosystem':'Ecosystem'}
