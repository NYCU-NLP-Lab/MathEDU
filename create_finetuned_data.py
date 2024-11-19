#!/usr/bin/env python
# coding: utf-8



import json
import numpy
import os
from datasets import load_dataset,concatenate_datasets




def id_map(responses):
    dataset = load_dataset("math_qa")
    mathqa = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
    for data in responses:
        data.update(mathqa[data['id']])
     
    return responses




with open(r'../finetune code/type3_data_woR_test_dataset.json', 'r', encoding='utf-8') as f:
    responses = [json.loads(line) for line in f]




#end to end 
#time_series_split
#wor




with open('dataset/time_series_split/train.json', 'r') as f:
    train_data=json.load(f)
with open('dataset/time_series_split/val.json', 'r') as f:
    val_data=json.load(f)
with open('dataset/time_series_split/test.json', 'r') as f:
    test_data=json.load(f)
    
id_map(train_data)
id_map(val_data)
id_map(test_data)




train=[]
val=[]
test=[]

for data in train_data:
    tmp={'messages':[]}
    tmp['messages'].append({'content':"You are a math teacher. According to the [Question], please indicate whether the [Student’s Answer] is correct or not. If the [Student’s Answer] is incorrect, identify where the student went wrong and provide explanations and advice. For correct answers, respond: The student's answer is correct. For incorrect answers, respond: The student's answer is incorrect. [wrong equation] ... [Teacher's explanations and advice] ...",'rule':'system'})
    if len(data['the_reason_why_student_cant_solve_en'])<3:
        tmp['messages'].append({'content':f"[Question] : {data['Problem']} options : {data['options']} [Student’s Answer] : {data['student_process']}",'rule':'user'})
    else:
        tmp['messages'].append({'content':f"[Question] : {data['Problem']} options : {data['options']} [Student’s Answer] : {data['student_process']} {data['the_reason_why_student_cant_solve_en']}",'rule':'user'})
    if data['correct_or_not']=='correct':
        tmp['messages'].append({'content':"The student's answer is correct.",'rule':'assistant'})
    else:
        ret="The student's answer is incorrect."
        for i in range(data['teacher_review']['error_counts']):
            if len(data['teacher_review']['error'][i]['error_equation'])>3:
                ret+=f" [Wrong equation {i+1}] : {data['teacher_review']['error'][i]['error_equation']} [Teacher's explanations and advice {i+1}] : {data['teacher_review']['error'][i]['teacher_advice_en']}"
            else:
                ret+=f" [Wrong equation {i+1}] : None [Teacher's explanations and advice {i+1}] : {data['teacher_review']['error'][i]['teacher_advice_en']}"
                
        tmp['messages'].append({'content':f"{ret}",'rule':'assistant'})
    train.append(tmp)

folder_path='finetuned_data'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    
with open('finetuned_data/end_to_end_time_series_split_wor_train.json', 'w') as f:
    json.dump(train, f)


for data in val_data:
    tmp={'messages':[]}
    tmp['messages'].append({'content':"You are a math teacher. According to the [Question], please indicate whether the [Student’s Answer] is correct or not. If the [Student’s Answer] is incorrect, identify where the student went wrong and provide explanations and advice. For correct answers, respond: The student's answer is correct. For incorrect answers, respond: The student's answer is incorrect. [wrong equation] ... [Teacher's explanations and advice] ...",'rule':'system'})
    if len(data['the_reason_why_student_cant_solve_en'])<3:
        tmp['messages'].append({'content':f"[Question] : {data['Problem']} options : {data['options']} [Student’s Answer] : {data['student_process']}",'rule':'user'})
    else:
        tmp['messages'].append({'content':f"[Question] : {data['Problem']} options : {data['options']} [Student’s Answer] : {data['student_process']} {data['the_reason_why_student_cant_solve_en']}",'rule':'user'})
    if data['correct_or_not']=='correct':
        tmp['messages'].append({'content':"The student's answer is correct.",'rule':'assistant'})
    else:
        ret="The student's answer is incorrect."
        for i in range(data['teacher_review']['error_counts']):
            if len(data['teacher_review']['error'][i]['error_equation'])>3:
                ret+=f" [Wrong equation {i+1}] : {data['teacher_review']['error'][i]['error_equation']} [Teacher's explanations and advice {i+1}] : {data['teacher_review']['error'][i]['teacher_advice_en']}"
            else:
                ret+=f" [Wrong equation {i+1}] : None [Teacher's explanations and advice {i+1}] : {data['teacher_review']['error'][i]['teacher_advice_en']}"
                
        tmp['messages'].append({'content':f"{ret}",'rule':'assistant'})
    val.append(tmp)

folder_path='finetuned_data'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    
with open('finetuned_data/end_to_end_time_series_split_wor_val.json', 'w') as f:
    json.dump(val, f)
    

for data in test_data:
    tmp={'messages':[]}
    tmp['messages'].append({'content':"You are a math teacher. According to the [Question], please indicate whether the [Student’s Answer] is correct or not. If the [Student’s Answer] is incorrect, identify where the student went wrong and provide explanations and advice. For correct answers, respond: The student's answer is correct. For incorrect answers, respond: The student's answer is incorrect. [wrong equation] ... [Teacher's explanations and advice] ...",'rule':'system'})
    if len(data['the_reason_why_student_cant_solve_en'])<3:
        tmp['messages'].append({'content':f"[Question] : {data['Problem']} options : {data['options']} [Student’s Answer] : {data['student_process']}",'rule':'user'})
    else:
        tmp['messages'].append({'content':f"[Question] : {data['Problem']} options : {data['options']} [Student’s Answer] : {data['student_process']} {data['the_reason_why_student_cant_solve_en']}",'rule':'user'})
    if data['correct_or_not']=='correct':
        tmp['messages'].append({'content':"The student's answer is correct.",'rule':'assistant'})
    else:
        ret="The student's answer is incorrect."
        for i in range(data['teacher_review']['error_counts']):
            if len(data['teacher_review']['error'][i]['error_equation'])>3:
                ret+=f" [Wrong equation {i+1}] : {data['teacher_review']['error'][i]['error_equation']} [Teacher's explanations and advice {i+1}] : {data['teacher_review']['error'][i]['teacher_advice_en']}"
            else:
                ret+=f" [Wrong equation {i+1}] : None [Teacher's explanations and advice {i+1}] : {data['teacher_review']['error'][i]['teacher_advice_en']}"
                
        tmp['messages'].append({'content':f"{ret}",'rule':'assistant'})
    test.append(tmp)

folder_path='finetuned_data'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    
with open('finetuned_data/end_to_end_time_series_split_wor_test.json', 'w') as f:
    json.dump(test, f)




#wr




train=[]
val=[]
test=[]

for data in train_data:
    tmp={'messages':[]}
    tmp['messages'].append({'content':"You are a math teacher. According to the [Question] and the [Rationale], please indicate whether the [Student’s Answer] is correct or not. If the [Student’s Answer] is incorrect, identify where the student went wrong and provide explanations and advice. For correct answers, respond: The student's answer is correct. For incorrect answers, respond: The student's answer is incorrect. [wrong equation] ... [Teacher's explanations and advice] ...",'rule':'system'})
    if len(data['the_reason_why_student_cant_solve_en'])<3:
        tmp['messages'].append({'content':f"[Question] : {data['Problem']} options : {data['options']} [Rationale] : {row['Rationale']} [Student’s Answer] : {data['student_process']}",'rule':'user'})
    else:
        tmp['messages'].append({'content':f"[Question] : {data['Problem']} options : {data['options']} [Rationale] : {row['Rationale']} [Student’s Answer] : {data['student_process']} {data['the_reason_why_student_cant_solve_en']}",'rule':'user'})
    if data['correct_or_not']=='correct':
        tmp['messages'].append({'content':"The student's answer is correct.",'rule':'assistant'})
    else:
        ret="The student's answer is incorrect."
        for i in range(data['teacher_review']['error_counts']):
            if len(data['teacher_review']['error'][i]['error_equation'])>3:
                ret+=f" [Wrong equation {i+1}] : {data['teacher_review']['error'][i]['error_equation']} [Teacher's explanations and advice {i+1}] : {data['teacher_review']['error'][i]['teacher_advice_en']}"
            else:
                ret+=f" [Wrong equation {i+1}] : None [Teacher's explanations and advice {i+1}] : {data['teacher_review']['error'][i]['teacher_advice_en']}"
                
        tmp['messages'].append({'content':f"{ret}",'rule':'assistant'})
    train.append(tmp)

folder_path='finetuned_data'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    
with open('finetuned_data/end_to_end_time_series_split_wr_train.json', 'w') as f:
    json.dump(train, f)


for data in val_data:
    tmp={'messages':[]}
    tmp['messages'].append({'content':"You are a math teacher. According to the [Question] and the [Rationale], please indicate whether the [Student’s Answer] is correct or not. If the [Student’s Answer] is incorrect, identify where the student went wrong and provide explanations and advice. For correct answers, respond: The student's answer is correct. For incorrect answers, respond: The student's answer is incorrect. [wrong equation] ... [Teacher's explanations and advice] ...",'rule':'system'})
    if len(data['the_reason_why_student_cant_solve_en'])<3:
        tmp['messages'].append({'content':f"[Question] : {data['Problem']} options : {data['options']} [Rationale] : {row['Rationale']} [Student’s Answer] : {data['student_process']}",'rule':'user'})
    else:
        tmp['messages'].append({'content':f"[Question] : {data['Problem']} options : {data['options']} [Rationale] : {row['Rationale']} [Student’s Answer] : {data['student_process']} {data['the_reason_why_student_cant_solve_en']}",'rule':'user'})
    if data['correct_or_not']=='correct':
        tmp['messages'].append({'content':"The student's answer is correct.",'rule':'assistant'})
    else:
        ret="The student's answer is incorrect."
        for i in range(data['teacher_review']['error_counts']):
            if len(data['teacher_review']['error'][i]['error_equation'])>3:
                ret+=f" [Wrong equation {i+1}] : {data['teacher_review']['error'][i]['error_equation']} [Teacher's explanations and advice {i+1}] : {data['teacher_review']['error'][i]['teacher_advice_en']}"
            else:
                ret+=f" [Wrong equation {i+1}] : None [Teacher's explanations and advice {i+1}] : {data['teacher_review']['error'][i]['teacher_advice_en']}"
                
        tmp['messages'].append({'content':f"{ret}",'rule':'assistant'})
    val.append(tmp)

folder_path='finetuned_data'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    
with open('finetuned_data/end_to_end_time_series_split_wr_val.json', 'w') as f:
    json.dump(val, f)
    

for data in test_data:
    tmp={'messages':[]}
    tmp['messages'].append({'content':"You are a math teacher. According to the [Question] and the [Rationale], please indicate whether the [Student’s Answer] is correct or not. If the [Student’s Answer] is incorrect, identify where the student went wrong and provide explanations and advice. For correct answers, respond: The student's answer is correct. For incorrect answers, respond: The student's answer is incorrect. [wrong equation] ... [Teacher's explanations and advice] ...",'rule':'system'})
    if len(data['the_reason_why_student_cant_solve_en'])<3:
        tmp['messages'].append({'content':f"[Question] : {data['Problem']} options : {data['options']} [Rationale] : {row['Rationale']} [Student’s Answer] : {data['student_process']}",'rule':'user'})
    else:
        tmp['messages'].append({'content':f"[Question] : {data['Problem']} options : {data['options']} [Rationale] : {row['Rationale']} [Student’s Answer] : {data['student_process']} {data['the_reason_why_student_cant_solve_en']}",'rule':'user'})
    if data['correct_or_not']=='correct':
        tmp['messages'].append({'content':"The student's answer is correct.",'rule':'assistant'})
    else:
        ret="The student's answer is incorrect."
        for i in range(data['teacher_review']['error_counts']):
            if len(data['teacher_review']['error'][i]['error_equation'])>3:
                ret+=f" [Wrong equation {i+1}] : {data['teacher_review']['error'][i]['error_equation']} [Teacher's explanations and advice {i+1}] : {data['teacher_review']['error'][i]['teacher_advice_en']}"
            else:
                ret+=f" [Wrong equation {i+1}] : None [Teacher's explanations and advice {i+1}] : {data['teacher_review']['error'][i]['teacher_advice_en']}"
                
        tmp['messages'].append({'content':f"{ret}",'rule':'assistant'})
    test.append(tmp)

folder_path='finetuned_data'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    
with open('finetuned_data/end_to_end_time_series_split_wr_test.json', 'w') as f:
    json.dump(test, f)




#leave-one-out
#wor




for n in range(1,7):
    with open(f'dataset/leave_one_out/student{n}/train.json', 'r') as f:
        train_data=json.load(f)
    with open(f'dataset/leave_one_out/student{n}/train.json', 'r') as f:
        val_data=json.load(f)
    with open(f'dataset/leave_one_out/student{n}/train.json', 'r') as f:
        test_data=json.load(f)

    id_map(train_data)
    id_map(val_data)
    id_map(test_data)

    train=[]
    val=[]
    test=[]

    for data in train_data:
        tmp={'messages':[]}
        tmp['messages'].append({'content':"You are a math teacher. According to the [Question], please indicate whether the [Student’s Answer] is correct or not. If the [Student’s Answer] is incorrect, identify where the student went wrong and provide explanations and advice. For correct answers, respond: The student's answer is correct. For incorrect answers, respond: The student's answer is incorrect. [wrong equation] ... [Teacher's explanations and advice] ...",'rule':'system'})
        if len(data['the_reason_why_student_cant_solve_en'])<3:
            tmp['messages'].append({'content':f"[Question] : {data['Problem']} options : {data['options']} [Student’s Answer] : {data['student_process']}",'rule':'user'})
        else:
            tmp['messages'].append({'content':f"[Question] : {data['Problem']} options : {data['options']} [Student’s Answer] : {data['student_process']} {data['the_reason_why_student_cant_solve_en']}",'rule':'user'})
        if data['correct_or_not']=='correct':
            tmp['messages'].append({'content':"The student's answer is correct.",'rule':'assistant'})
        else:
            ret="The student's answer is incorrect."
            for i in range(data['teacher_review']['error_counts']):
                if len(data['teacher_review']['error'][i]['error_equation'])>3:
                    ret+=f" [Wrong equation {i+1}] : {data['teacher_review']['error'][i]['error_equation']} [Teacher's explanations and advice {i+1}] : {data['teacher_review']['error'][i]['teacher_advice_en']}"
                else:
                    ret+=f" [Wrong equation {i+1}] : None [Teacher's explanations and advice {i+1}] : {data['teacher_review']['error'][i]['teacher_advice_en']}"

            tmp['messages'].append({'content':f"{ret}",'rule':'assistant'})
        train.append(tmp)

    folder_path='finetuned_data'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(f'finetuned_data/end_to_end_leave_one_out_student{n}_wor_train.json', 'w') as f:
        json.dump(train, f)


    for data in val_data:
        tmp={'messages':[]}
        tmp['messages'].append({'content':"You are a math teacher. According to the [Question], please indicate whether the [Student’s Answer] is correct or not. If the [Student’s Answer] is incorrect, identify where the student went wrong and provide explanations and advice. For correct answers, respond: The student's answer is correct. For incorrect answers, respond: The student's answer is incorrect. [wrong equation] ... [Teacher's explanations and advice] ...",'rule':'system'})
        if len(data['the_reason_why_student_cant_solve_en'])<3:
            tmp['messages'].append({'content':f"[Question] : {data['Problem']} options : {data['options']} [Student’s Answer] : {data['student_process']}",'rule':'user'})
        else:
            tmp['messages'].append({'content':f"[Question] : {data['Problem']} options : {data['options']} [Student’s Answer] : {data['student_process']} {data['the_reason_why_student_cant_solve_en']}",'rule':'user'})
        if data['correct_or_not']=='correct':
            tmp['messages'].append({'content':"The student's answer is correct.",'rule':'assistant'})
        else:
            ret="The student's answer is incorrect."
            for i in range(data['teacher_review']['error_counts']):
                if len(data['teacher_review']['error'][i]['error_equation'])>3:
                    ret+=f" [Wrong equation {i+1}] : {data['teacher_review']['error'][i]['error_equation']} [Teacher's explanations and advice {i+1}] : {data['teacher_review']['error'][i]['teacher_advice_en']}"
                else:
                    ret+=f" [Wrong equation {i+1}] : None [Teacher's explanations and advice {i+1}] : {data['teacher_review']['error'][i]['teacher_advice_en']}"

            tmp['messages'].append({'content':f"{ret}",'rule':'assistant'})
        val.append(tmp)

    folder_path='finetuned_data'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(f'finetuned_data/end_to_end_leave_one_out_student{n}_wor_val.json', 'w') as f:
        json.dump(val, f)


    for data in test_data:
        tmp={'messages':[]}
        tmp['messages'].append({'content':"You are a math teacher. According to the [Question], please indicate whether the [Student’s Answer] is correct or not. If the [Student’s Answer] is incorrect, identify where the student went wrong and provide explanations and advice. For correct answers, respond: The student's answer is correct. For incorrect answers, respond: The student's answer is incorrect. [wrong equation] ... [Teacher's explanations and advice] ...",'rule':'system'})
        if len(data['the_reason_why_student_cant_solve_en'])<3:
            tmp['messages'].append({'content':f"[Question] : {data['Problem']} options : {data['options']} [Student’s Answer] : {data['student_process']}",'rule':'user'})
        else:
            tmp['messages'].append({'content':f"[Question] : {data['Problem']} options : {data['options']} [Student’s Answer] : {data['student_process']} {data['the_reason_why_student_cant_solve_en']}",'rule':'user'})
        if data['correct_or_not']=='correct':
            tmp['messages'].append({'content':"The student's answer is correct.",'rule':'assistant'})
        else:
            ret="The student's answer is incorrect."
            for i in range(data['teacher_review']['error_counts']):
                if len(data['teacher_review']['error'][i]['error_equation'])>3:
                    ret+=f" [Wrong equation {i+1}] : {data['teacher_review']['error'][i]['error_equation']} [Teacher's explanations and advice {i+1}] : {data['teacher_review']['error'][i]['teacher_advice_en']}"
                else:
                    ret+=f" [Wrong equation {i+1}] : None [Teacher's explanations and advice {i+1}] : {data['teacher_review']['error'][i]['teacher_advice_en']}"

            tmp['messages'].append({'content':f"{ret}",'rule':'assistant'})
        test.append(tmp)

    folder_path='finetuned_data'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(f'finetuned_data/end_to_end_leave_one_out_student{n}_wor_test.json', 'w') as f:
        json.dump(test, f)




#wr




for n in range(1,7):
    with open(f'dataset/leave_one_out/student{n}/train.json', 'r') as f:
        train_data=json.load(f)
    with open(f'dataset/leave_one_out/student{n}/train.json', 'r') as f:
        val_data=json.load(f)
    with open(f'dataset/leave_one_out/student{n}/train.json', 'r') as f:
        test_data=json.load(f)

    id_map(train_data)
    id_map(val_data)
    id_map(test_data)

    train=[]
    val=[]
    test=[]

    for data in train_data:
        tmp={'messages':[]}
        tmp['messages'].append({'content':"You are a math teacher. According to the [Question] and the [Rationale], please indicate whether the [Student’s Answer] is correct or not. If the [Student’s Answer] is incorrect, identify where the student went wrong and provide explanations and advice. For correct answers, respond: The student's answer is correct. For incorrect answers, respond: The student's answer is incorrect. [wrong equation] ... [Teacher's explanations and advice] ...",'rule':'system'})
        if len(data['the_reason_why_student_cant_solve_en'])<3:
            tmp['messages'].append({'content':f"[Question] : {data['Problem']} options : {data['options']} [Rationale] : {row['Rationale']} [Student’s Answer] : {data['student_process']}",'rule':'user'})
        else:
            tmp['messages'].append({'content':f"[Question] : {data['Problem']} options : {data['options']} [Rationale] : {row['Rationale']} [Student’s Answer] : {data['student_process']} {data['the_reason_why_student_cant_solve_en']}",'rule':'user'})
        if data['correct_or_not']=='correct':
            tmp['messages'].append({'content':"The student's answer is correct.",'rule':'assistant'})
        else:
            ret="The student's answer is incorrect."
            for i in range(data['teacher_review']['error_counts']):
                if len(data['teacher_review']['error'][i]['error_equation'])>3:
                    ret+=f" [Wrong equation {i+1}] : {data['teacher_review']['error'][i]['error_equation']} [Teacher's explanations and advice {i+1}] : {data['teacher_review']['error'][i]['teacher_advice_en']}"
                else:
                    ret+=f" [Wrong equation {i+1}] : None [Teacher's explanations and advice {i+1}] : {data['teacher_review']['error'][i]['teacher_advice_en']}"

            tmp['messages'].append({'content':f"{ret}",'rule':'assistant'})
        train.append(tmp)

    folder_path='finetuned_data'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(f'finetuned_data/end_to_end_leave_one_out_student{n}_wr_train.json', 'w') as f:
        json.dump(train, f)


    for data in val_data:
        tmp={'messages':[]}
        tmp['messages'].append({'content':"You are a math teacher. According to the [Question] and the [Rationale], please indicate whether the [Student’s Answer] is correct or not. If the [Student’s Answer] is incorrect, identify where the student went wrong and provide explanations and advice. For correct answers, respond: The student's answer is correct. For incorrect answers, respond: The student's answer is incorrect. [wrong equation] ... [Teacher's explanations and advice] ...",'rule':'system'})
        if len(data['the_reason_why_student_cant_solve_en'])<3:
            tmp['messages'].append({'content':f"[Question] : {data['Problem']} options : {data['options']} [Rationale] : {row['Rationale']} [Student’s Answer] : {data['student_process']}",'rule':'user'})
        else:
            tmp['messages'].append({'content':f"[Question] : {data['Problem']} options : {data['options']} [Rationale] : {row['Rationale']} [Student’s Answer] : {data['student_process']} {data['the_reason_why_student_cant_solve_en']}",'rule':'user'})
        if data['correct_or_not']=='correct':
            tmp['messages'].append({'content':"The student's answer is correct.",'rule':'assistant'})
        else:
            ret="The student's answer is incorrect."
            for i in range(data['teacher_review']['error_counts']):
                if len(data['teacher_review']['error'][i]['error_equation'])>3:
                    ret+=f" [Wrong equation {i+1}] : {data['teacher_review']['error'][i]['error_equation']} [Teacher's explanations and advice {i+1}] : {data['teacher_review']['error'][i]['teacher_advice_en']}"
                else:
                    ret+=f" [Wrong equation {i+1}] : None [Teacher's explanations and advice {i+1}] : {data['teacher_review']['error'][i]['teacher_advice_en']}"

            tmp['messages'].append({'content':f"{ret}",'rule':'assistant'})
        val.append(tmp)

    folder_path='finetuned_data'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(f'finetuned_data/end_to_end_leave_one_out_student{n}_wr_val.json', 'w') as f:
        json.dump(val, f)


    for data in test_data:
        tmp={'messages':[]}
        tmp['messages'].append({'content':"You are a math teacher. According to the [Question] and the [Rationale], please indicate whether the [Student’s Answer] is correct or not. If the [Student’s Answer] is incorrect, identify where the student went wrong and provide explanations and advice. For correct answers, respond: The student's answer is correct. For incorrect answers, respond: The student's answer is incorrect. [wrong equation] ... [Teacher's explanations and advice] ...",'rule':'system'})
        if len(data['the_reason_why_student_cant_solve_en'])<3:
            tmp['messages'].append({'content':f"[Question] : {data['Problem']} options : {data['options']} [Rationale] : {row['Rationale']} [Student’s Answer] : {data['student_process']}",'rule':'user'})
        else:
            tmp['messages'].append({'content':f"[Question] : {data['Problem']} options : {data['options']} [Rationale] : {row['Rationale']} [Student’s Answer] : {data['student_process']} {data['the_reason_why_student_cant_solve_en']}",'rule':'user'})
        if data['correct_or_not']=='correct':
            tmp['messages'].append({'content':"The student's answer is correct.",'rule':'assistant'})
        else:
            ret="The student's answer is incorrect."
            for i in range(data['teacher_review']['error_counts']):
                if len(data['teacher_review']['error'][i]['error_equation'])>3:
                    ret+=f" [Wrong equation {i+1}] : {data['teacher_review']['error'][i]['error_equation']} [Teacher's explanations and advice {i+1}] : {data['teacher_review']['error'][i]['teacher_advice_en']}"
                else:
                    ret+=f" [Wrong equation {i+1}] : None [Teacher's explanations and advice {i+1}] : {data['teacher_review']['error'][i]['teacher_advice_en']}"

            tmp['messages'].append({'content':f"{ret}",'rule':'assistant'})
        test.append(tmp)

    folder_path='finetuned_data'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(f'finetuned_data/end_to_end_leave_one_out_student{n}_wr_test.json', 'w') as f:
        json.dump(test, f)

