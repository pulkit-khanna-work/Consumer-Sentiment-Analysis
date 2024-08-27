# Domain-Specific Consumer Sentiment Analysis with Intent & Entity Extraction

Aim: To perform intent and sentiment analysis for retail products with an overall accuracy of atleast 80%. We're carefully choosing between RoBERTa and LLAMA 8B models for our sentiment analysis, prioritizing sustainability and minimizing our carbon footprint by avoiding larger LLMs where possible.

## Approach:
We perform intent detection using a regex dictionary of domain and subdomain taxonomy, based on customer perspectives, segregating the sentences into their appropiate categories followed by a sentiment model.
We have trained/finetuned various models such as VADER, RoBERTa, DeBERTa, LLama, Mistral etc respectively for sentiment analysis and are using Llama 8B and RoBERTa currently.

## Sections - 
* main.py - The main file for running inference and uses helper functions from intent and sentiment analysis folders
1) Intent Analysis w/ Utils - Intent classification and splitting, competitor comparative case handling, subdomain analysis and helper utils
2) Sentiment Analysis w/ Utils - Data preprocessing, pros con handling, finetuning utils, main inferencing functions and other configurations related to the task
3) Sentence Generation using SLM - Similar sentence, opposite sentence, paraphrasing sentence over clustered data over regex
4) Finetuning (RoBERTa, Llama 8B Model) - RoBERTa & Llama model finetuning using unsloth over custom batching for varied examples clusters per batch for sentiment analysis [Uses helper functions from intent and sentiment analysis folders]

## Relevant Papers:
A. Joshy and S. Sundar, "Analyzing the Performance of Sentiment Analysis using BERT, DistilBERT, and RoBERTa," 2022 IEEE International Power and Renewable Energy Conference (IPRECON), Kollam, India, 2022, pp. 1-6, doi: 10.1109/IPRECON55716.2022.10059542.

Comparison of VADER and Pre-Trained RoBERTa: A Sentiment Analysis Application, Authors: Erwe, Linda LU and Wang, Xin LU

Jakub Šmíd, Pavel Priban, and Pavel Kral. 2024. LLaMA-Based Models for Aspect-Based Sentiment Analysis. In Proceedings of the 14th Workshop on Computational Approaches to Subjectivity, Sentiment, & Social Media Analysis, pages 63–70, Bangkok, Thailand. Association for Computational Linguistics.
