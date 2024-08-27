#Regex for finding comparative words
Comparative_words = [
    r'\bfast[esr]*\b',
    r'\bslow[esr]*\b',
    r'\bstrong[esr]*\b',
    r'\bheav[eir]*\b',
    r'(?<!bad )(?<!bright )(?<!dim )(?<!red )(?<!green )\blighte?r?\b(?!.*painting)',
    
    r'\blesse?r?\b'
    r'\bthan\b',
    r'\bthen\b',
    r'\bsimilar\b',
    r'\bcontrast\b',
    r'\brelativel?y?\b',
    r'\bequivalent\b',
    r'\bcompared?\b',
    r'\bComparable\b',
    r'\bcomparison\b',
    r'\bcomparatively\b',
    r'\bcontrastl?y?\b',
    r'\btransition\b',
    r'\bunlike\b'
    r'\binferior\b'
    # r'\bworst\b',
    r'\bworser\b',
    # r'\bbest\b', #makes indirect
    r'(?<!I )\blike\b',
    r'\bas\b(?!\s*part\s*of\s*promotion\b)',
    r'\bfrom\b']

comp_sep = ['\n', '.','!','?','<<TB4>>']
#Note: if both are negative ?
#'Iphone price is too expensive but this phones price is even higher at no extra feature'


#If A is __ than B then sentiment found is for A and reverse of such is B
# Iphone has a better camera than this phone:  +1 Iphone, -1 our model
# Iphone is better :Default our model
# previous phone  :Default our model

# both default cases check

words_where_senti_for_first_and_second_is_reverse = [
    r'\bthan\b',
    r'\bcompared?\b',
    r'\bComparable\b',
    r'\bcomparison\b',
    r'\bcomparatively\b',
    r'\bcontrastl?y?\b',
    r'\brelativel?y?\b',
    r'\bunlike\b'
    r'\binferior\b'
    r'\bwell\s*behind\b',
    r'\blesse?r?\b'
]

words_where_senti_remains_same_for_both=[
    r'\bequivalent\b',
    r'(?<!I )\blike\b(?!\s*it\b|\s*me\b|\s*him\b|\s*her\b|\s*them\b|\s*us\b)',
    r'\bsimilar\b',
]
#Primary is our model and others are neutral entity
eternal_our_model_comp_words=[
    r'\btransition\b',
    r'\btransferring\b'
    r'\bswitching\b'
]
#transition ones will remain same
#similar to than but instead of second entity being reverse sentiment it is neutral
from_word=[
    r'^(?:(?!\btransition\b|\btransferring\b|\bswitching\b).)*\bfrom\b',
]
#but only for model mentioned general regex then our model else the model in question
#if comp model coming as best/worst then not to take it to main file else if our model then if empty fielf for intent then update it in main
words_making_it_only_indirect=[
    r'\bworst\b',
    r'\bbest\b'
]

words_with_special_cond=[
    r'\bas\b', #is not as then 1st entity's intent is reversed for sencond or if 'is as' without any not/ then both have same senti
]


rem_wrds=[
    #heavier/lighter/faster/slower can come with than so no issues
    r'\bfast[est]*\b',
    r'\bslow[est]*\b',
    r'\bstrong[est]*\b',
    r'\bheav[esit]*\b',
    r'\blight[est]*\b',

    r'\bmore\b',
    r'\bthen\b',
    r'\bworser\b',
]

# Iphones are best in terms of their camera
