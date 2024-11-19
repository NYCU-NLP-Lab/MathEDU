#!/usr/bin/env python
# coding: utf-8


import json
import openai
from tqdm import tqdm
import random
import time
from datasets import load_dataset,concatenate_datasets

openai.api_key = 'your_api_key'




def id_map(responses):
    dataset = load_dataset("math_qa")
    mathqa = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
    for data in responses:
        data.update(mathqa[data['id']])
     
    return responses




def create_prompt_o1(Type,examples,test):
    prompt=[]
    if Type=='woR':
        prompt.append({"role": "user", "content":"You are a math teacher. According to the [Question], please indicate whether the [Student’s Answer] is correct or not. If the [Student’s Answer] is incorrect, identify where the student went wrong and provide explanations and advice. For correct answers, respond: The student's answer is correct. For incorrect answers, respond: The student's answer is incorrect. [wrong equation] ... [Teacher's explanations and advice] ..."})
        for row in examples:
            if len(row['the_reason_why_student_cant_solve_en'])<3:
                prompt.append({ "role": "user", "content":f"[Question] : {row['Problem']} options : {row['options']} [Student’s Answer] : {row['student_process']}"})
            else:
                prompt.append({ "role": "user", "content":f"[Question] : {row['Problem']} options : {row['options']} [Student’s Answer] : {row['student_process']} {row['the_reason_why_student_cant_solve_en']}"})                
            
            if row['correct_or_not']=='correct':
                prompt.append({"role": "assistant", "content":"The student's answer is correct."})
            else:
                tmp = "The student's answer is incorrect."
                for i in range(row['teacher_review']['error_counts']):
                    if len(row['teacher_review']['error'][i]['error_equation'])>3:
                        tmp+=f" [Wrong equation {i+1}] : {row['teacher_review']['error'][i]['error_equation']} [Teacher's explanations and advice {i+1}] : {row['teacher_review']['error'][i]['teacher_advice_en']}"
                    else:
                        tmp+=f" [Wrong equation {i+1}] : None [Teacher's explanations and advice {i+1}] : {row['teacher_review']['error'][i]['teacher_advice_en']}"
                prompt.append({"role": "assistant", "content":tmp})
        if len(test['the_reason_why_student_cant_solve_en'])<3:
            prompt.append({"role": "user", "content":f"[Question] : {test['Problem']} options : {test['options']} [Student’s Answer] : {test['student_process']}"})
        else:
            prompt.append({"role": "user", "content":f"[Question] : {test['Problem']} options : {test['options']} [Student’s Answer] : {test['student_process']} {test['the_reason_why_student_cant_solve_en']}"})

                          
    elif Type=='wR':
        prompt.append({"role": "user", "content":"You are a math teacher. According to the [Question] and the [Rationale], please indicate whether the [Student’s Answer] is correct or not. If the [Student’s Answer] is incorrect, identify where the student went wrong and provide explanations and advice. For correct answers, respond: The student's answer is correct. For incorrect answers, respond: The student's answer is incorrect. [wrong equation] ... [Teacher's explanations and advice] ..."})
        for row in examples:
            if len(row['the_reason_why_student_cant_solve_en'])<3:
                prompt.append({"role": "user", "content":f"[Question] : {row['Problem']} options : {row['options']}  [Rationale] : {row['Rationale']} [Student’s Answer] : {row['student_process']}"})
            else:
                prompt.append({"role": "user", "content":f"[Question] : {row['Problem']} options : {row['options']}  [Rationale] : {row['Rationale']} [Student’s Answer] : {row['student_process']} {row['the_reason_why_student_cant_solve_en']}"})                
            
            if row['correct_or_not']=='correct':
                prompt.append({"role": "assistant", "content":"The student's answer is correct."})
            else:
                tmp = "The student's answer is incorrect."
                for i in range(row['teacher_review']['error_counts']):
                    if len(row['teacher_review']['error'][i]['error_equation'])>3:
                        tmp+=f" [Wrong equation {i+1}] : {row['teacher_review']['error'][i]['error_equation']} [Teacher's explanations and advice {i+1}] : {row['teacher_review']['error'][i]['teacher_advice_en']}"
                    else:
                        tmp+=f" [Wrong equation {i+1}] : None [Teacher's explanations and advice {i+1}] : {row['teacher_review']['error'][i]['teacher_advice_en']}"
                prompt.append({"role": "assistant", "content":tmp})
                    
        if len(test['the_reason_why_student_cant_solve_en'])<3:
            prompt.append({"role": "user", "content":f"[Question] : {test['Problem']} options : {test['options']}  [Rationale] : {row['Rationale']} [Student’s Answer] : {test['student_process']}"})
        else:
            prompt.append({"role": "user", "content":f"[Question] : {test['Problem']} options : {test['options']}  [Rationale] : {row['Rationale']} [Student’s Answer] : {test['student_process']} {test['the_reason_why_student_cant_solve_en']}"})         
    
    return prompt




def get_examples(train_data,problem):
    examples=[]
    
    count=0
    for element in train_data:
        if element['correct_or_not']=='correct' and element['student_id']==problem['student_id']:
            examples.append(element)
            count+=1
        if count==3:
            break

    for element in train_data:
        if element['correct_or_not']=='wrong' and element['student_id']==problem['student_id']:
            if element['teacher_review']['error'][0]['error_equation']=='None':
                examples.append(element)
                break

    for element in train_data:
        if element['correct_or_not']=='wrong' and element['student_id']==problem['student_id']:
            if element['teacher_review']['error'][0]['error_equation']!='None':
                if element['teacher_review']['error_counts']==1:
                    examples.append(element)
                    break

    for element in train_data:
        if element['correct_or_not']=='wrong' and element['student_id']==problem['student_id']:
            if element['teacher_review']['error'][0]['error_equation']!='None':
                if element['teacher_review']['error_counts']>1:
                    examples.append(element)
                break

    return examples



#time_series_split
##wor




with open('dataset/time_series_split/train.json', 'r') as f:
    train_data=json.load(f)
    
random.seed(42)
random.shuffle(train_data)

with open(r'dataset/time_series_split/test.json', 'r') as f:
    test_data=json.load(f)
    
id_map(train_data)
id_map(test_data)

for element in tqdm(test_data):
    if 'gpt_o1_mini_response' in element.keys():
        continue
    examples = get_examples(train_data,element)
    prompt=create_prompt_o1("woR",examples,element)
    response = openai.ChatCompletion.create(
                model="o1-mini",
                messages=prompt,
    )
    element['o1_mini_response'] = response["choices"][0]['message']['content']
    
    time.sleep(0.5)

with open('model_response/o1_mini_grading_result_time_series_split_woR.json', 'w') as f: 
    json.dump(test_data, f)




#time_series_split
##wr



with open('dataset/time_series_split/train.json', 'r') as f:
    train_data=json.load(f)
    
random.seed(42)
random.shuffle(train_data)

with open(r'dataset/time_series_split/test.json', 'r') as f:
    test_data=json.load(f)
    
id_map(train_data)
id_map(test_data)

for element in tqdm(test_data):
    if 'gpt_o1_mini_response' in element.keys():
        continue
    examples = get_examples(train_data,element)
    prompt=create_prompt_o1("wR",examples,element)
    response = openai.ChatCompletion.create(
                model="o1-mini",
                messages=prompt,
    )
    element['o1_mini_response'] = response["choices"][0]['message']['content']
    
    time.sleep(0.5)

with open('model_response/o1_mini_grading_result_time_series_split_wR.json', 'w') as f: 
    json.dump(test_data, f)

