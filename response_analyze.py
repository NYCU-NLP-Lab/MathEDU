#!/usr/bin/env python
# coding: utf-8


import json
import numpy
from rouge import Rouge 
from bert_score import score
from datasets import load_dataset,concatenate_datasets
import copy
from rouge import Rouge 
rouge = Rouge()



DISTANCE_PENALTY=127


def id_map(responses):
    dataset = load_dataset("math_qa")
    mathqa = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
    for data in responses:
        data.update(mathqa[data['id']])
     
    return responses



def judge(response):
    if "The student's answer is correct." in response:
        return True
    elif " is the correct answer." in response:
        return True
    elif " is correct." in response:
        return True
    elif " correct." in response:
        return True
    elif "**correct**." in response:
        return True
    elif "the student's answer is correct." in response:
        return True
    elif "The student's answer is incorrect." in response:
        return False
    elif " wrong " in response:
        return False
    elif "**incorrect**." in response:
        return False
    elif "" == response.strip():
        return -1
    else:
        return False



def get_wrong_equation(response):
    equation=[]

    if "[Wrong equation] :" in response:
        equation.append(response.split(f"[Wrong equation] :")[1].split(f" [Teacher")[0].strip())
        return equation
    
    if "[Wrong equation]" in response:
        equation.append(response.split(f"[Wrong equation]")[1].split(f" [Teacher")[0].strip())
        return equation
            
    if "**Wrong equation:**" in response:
        equation.append(response.split(f"\[")[1].split(f"**Teacher's explanations and advice:**")[0].split('\]')[0].strip())
        print(response.split(f"\[")[1].split(f"**Teacher's explanations and advice:**")[0].split('\]')[0].strip())
        print('===============================')
        return equation
    
    for i in range(1,5):
        try:
            equation.append(response.split(f"[Wrong equation {i}] :")[1].split(f" [Teacher")[0].strip())
        except:
            break
             
    return equation



def calculate_distance(answer,response_equlist):
    student_process = answer['student_process']
    ground_truth = []
    groundtruth_posision=[] # [start_pos, end_pos]
    response_posision=[] # [start_pos, end_pos]
    for err in answer['teacher_review']['error']:
        ground_truth.append(err['error_equation'].strip())
        
    for equa in ground_truth:
        if equa == 'None':
            groundtruth_posision.append([-1,-1])
        else:
            groundtruth_posision.append([student_process.find(equa),student_process.find(equa)+len(equa)])
            
    for equa in response_equlist:
        if equa == 'None':
            response_posision.append([-1,-1])
        else:
            response_posision.append([student_process.find(equa),student_process.find(equa)+len(equa)])
    
    tmp_distance=0
    ret_dis=10000 
    for equa_ground in groundtruth_posision:
        for equa_resp in response_posision:
            tmp_distance=0
            #both None
            if(equa_ground==[-1,-1] and equa_resp==[-1,-1]):
                ret_dis=0
                return ret_dis
            
            # only one none
            elif (equa_ground==[-1,-1] and equa_resp!=[-1,-1] or equa_ground!=[-1,-1] and equa_resp==[-1,-1]):
                ret_dis = min(DISTANCE_PENALTY,ret_dis)
            
            # perfect match
            elif(equa_ground[0]==equa_resp[0] and equa_ground[1]==equa_resp[1]):
                ret_dis=0
                return ret_dis
            
            #      gggg 
            # pppp
            elif(equa_resp[1]<=equa_ground[0]):
                tmp_distance=equa_ground[0]-equa_resp[0]
                ret_dis=min(tmp_distance,ret_dis)
            
            # gggg   
            #      pppp
            elif(equa_ground[1]<=equa_resp[0]):
                tmp_distance=equa_resp[0]-equa_ground[0]
                ret_dis=min(tmp_distance,ret_dis)
            
            
            # gggggggggggg
            #     pppp
            elif(equa_ground[0]<=equa_resp[0] and equa_resp[1]<=equa_ground[1]):
                tmp_distance=equa_resp[0]-equa_ground[0] + equa_ground[1]-equa_resp[1] 
                ret_dis=min(tmp_distance,ret_dis)
                
            #     gggg
            # pppppppppppp 
            elif(equa_resp[0]<=equa_ground[0] and equa_ground[1]<=equa_resp[1]):
                tmp_distance=equa_ground[0]-equa_resp[0] + equa_resp[1]-equa_ground[1] 
                ret_dis=min(tmp_distance,ret_dis)
            
            #  ggggggg
            #    ppppppp
            elif(equa_ground[0]<=equa_resp[0] and equa_resp[1]<=equa_ground[1]):
                tmp_distance=equa_resp[0]-equa_ground[0] + equa_resp[1]-equa_ground[1] 
                ret_dis=min(tmp_distance,ret_dis)
                
            #    ggggggg
            #  ppppppp
            elif(equa_resp[0]<=equa_ground[0] and equa_ground[1]<=equa_resp[1]):
                tmp_distance=equa_ground[0]-equa_resp[0] + equa_resp[1]-equa_ground[1] 
                ret_dis=min(tmp_distance,ret_dis)
                
            else:
                ret_dis = min(DISTANCE_PENALTY,ret_dis)
                #print('error')
                
    return ret_dis         



def get_teacher_advice(response):
    advice=""
    for i in range(1,5):
        try:
            tmp=response.split(f"[Teacher's explanations and advice {i}] :")[1].split(f"[Wrong")[0]
            advice+=tmp

        except:
            if advice !='':
                return advice
            if "**Teacher's explanations and advice:**" in response:
                advice+=response.split("**Teacher's explanations and advice:**")[1]
            elif "**Teacher's explanations and advice**:" in response:
                advice+=response.split("**Teacher's explanations and advice**:")[1]
            elif r"**\[Teacher's Explanations and Advice\]:**" in response:
                advice+=response.split(r"**\[Teacher's Explanations and Advice\]:**")[1]
            elif "**Teacher's explanations and advice:]**" in response:
                advice+=response.split("**Teacher's explanations and advice:]**")[1]
            elif "**Teacher's Explanations and Advice:**" in response:
                advice+=response.split("**Teacher's Explanations and Advice:**")[1]
            elif "**[Teacher's explanations and advice 1]:**" in response:
                advice+=response.split("**[Teacher's explanations and advice 1]:**")[1]
            elif "**[Teacher's explanations and advice 1]**:" in response:
                advice+=response.split("**[Teacher's explanations and advice 1]**:")[1]
            elif "**[Teacher's Explanations and Advice]**" in response:
                advice+=response.split("**[Teacher's Explanations and Advice]**")[1] 
            elif "**[Teacher's Explanations and Advice]:**" in response:
                advice+=response.split("**[Teacher's Explanations and Advice]:**")[1]
            elif "**Explanation:**" in response:
                advice+=response.split("**Explanation:**")[1]
            elif "[Teacher's Explanations and Advice]:**" in response:
                advice+=response.split("[Teacher's Explanations and Advice]:**")[1]
            elif "[Teacher's Explanations and Advice]**:" in response:
                advice+=response.split("[Teacher's Explanations and Advice]**:")[1]
            elif "[Teacher's explanations and advice ] :" in response:
                advice+=response.split("[Teacher's explanations and advice ] :")[1]
            elif r"[Teacher's Explanations and Advice\]:**" in response:
                advice+=response.split(r"[Teacher's Explanations and Advice\]:**")[1]    
            
            elif "**Explanation of the Mistake:**" in response:
                advice+=response.split("**Explanation of the Mistake:**")[1]
            elif "[Teacher's explanations and advice 1]**:" in response:
                advice+=response.split("[Teacher's explanations and advice 1]**:")[1]
                
            elif "[Teacher's explanations and advice 1]:**" in response:
                advice+=response.split("[Teacher's explanations and advice 1]:**")[1]
            elif "[Teacher's explanations and advice 1]:**" in response:
                advice+=response.split("[Teacher's explanations and advice 1]:**")[1]
                  
            elif "[Teacher's Explanations and Advice]**" in response:
                advice+=response.split("[Teacher's Explanations and Advice]**")[1]
            elif "[Teacher's explanations and advice]" in response:
                advice+=response.split("[Teacher's explanations and advice]")[1]
            return advice
            
    return advice



print('Correctness Identification')
print('time series split\n')

with open(r'model_response/time_series_split_correctness_identification.json', 'r') as f:
    responses=json.load(f)
        
id_map(responses)     
        
category_init={}
# [all, correct]
category_init['all'] = [0,0]
for dic in responses:
    if dic['category'] not in category_init:
        category_init[dic['category']]=[0,0]
    if len(category_init)==7:
        break
        
correct_dic={}
for key, value in responses[0].items():
    if 'wr' not in key and 'wor' not in key:
        continue
    else:
        correct_dic[key]={k: v[:] for k, v in category_init.items()}


for dic in responses:
    for key, value in dic.items():
        if 'wr' not in key and 'wor' not in key:
            continue
        else:
            correct_dic[key]['all'][0]+=1
            correct_dic[key][dic['category']][0]+=1
            if dic['correct_or_not']=='correct' and judge(value)==1 or dic['correct_or_not']=='wrong' and judge(value)==0:
                correct_dic[key]['all'][1]+=1
                correct_dic[key][dic['category']][1]+=1
            else:
                if key=='gpt35_wor':
                    #print(value)
                    continue
                continue

for key, value in correct_dic.items():
    print(f'{key} accuracy :')
    for category, count in value.items():
        print(f'{category} : {count[1]/count[0]*100:.2f}%')
    print()


print('leave one out\n')

tmp_dic={}
tmp_dic['all']=0
tmp_dic['general']=0
tmp_dic['gain']=0
tmp_dic['physics']=0
tmp_dic['geometry']=0
tmp_dic['probability']=0
tmp_dic['other']=0

sum_dic={}

for key, value in responses[0].items():
    if 'wr' not in key and 'wor' not in key:
        continue
    else:
        sum_dic[key]={k: v for k, v in tmp_dic.items()}

for n in range(1,7):
    with open(f'model_response/leave_one_out_student{n}_correctness_identification.json', 'r') as f:
        responses=json.load(f)
    id_map(responses)    
    
                    
    category_init={}
    # [all, correct]
    category_init['all'] = [0,0]
    for dic in responses:
        if dic['category'] not in category_init:
            category_init[dic['category']]=[0,0]
        if len(category_init)==7:
            break

    correct_dic={}
    for key, value in responses[0].items():
        if 'wr' not in key and 'wor' not in key:
            continue
        else:
            correct_dic[key]={k: v[:] for k, v in category_init.items()}


    for dic in responses:
        for key, value in dic.items():
            if 'wr' not in key and 'wor' not in key:
                continue
            else:
                correct_dic[key]['all'][0]+=1
                correct_dic[key][dic['category']][0]+=1
                if dic['correct_or_not']=='correct' and judge(value)==1 or dic['correct_or_not']=='wrong' and judge(value)==0:
                    correct_dic[key]['all'][1]+=1
                    correct_dic[key][dic['category']][1]+=1
                else:
                    if key=='gpt35_wor':
                        #print(value)
                        continue
                    continue
    print(f'student {n}')
    for key, value in correct_dic.items():
        print(f'{key} accuracy :')
        for category, count in value.items():
            print(f'{category} : {count[1]/count[0]*100:.2f}%')
            sum_dic[key][category]+=count[1]/count[0]*100
        print()
            
print('Over all')
for key, value in sum_dic.items():
    print(f'{key} accuracy :')
    for category, count in value.items():
        print(f'{category:20} : {count/6:.2f}%')
    print()



print('Problem-Solving Error Identification')
print('time series split\n')

with open(r'model_response/time_series_split_problem-solving_error_identification.json', 'r') as f:
    responses=json.load(f)

correct_dic={}
for key, value in responses[0].items():
    if 'wr' not in key and 'wor' not in key:
        continue
    else:
        # [ground turth wrong amount, em count, dis]
        correct_dic[key]=[0,0,0]
        
for dic in responses:
    for key, value in dic.items():
        if 'wr' not in key and 'wor' not in key:
            continue
        else:
            if dic['correct_or_not']=='wrong':
                correct_dic[key][0]+=1
                if value != '' and value != "The student's answer doesn't contain wrong equations.":
                    if key == 'o1_mini_wor':
                        response_equa_list=[value]
                    else:
                        response_equa_list = get_wrong_equation(value)
                    if len(response_equa_list)==0:
                        correct_dic[key][2]+=DISTANCE_PENALTY
                        continue
                    ground_truth_teacher=[]
                    for error in dic['teacher_review']['error']:
                        ground_truth_teacher.append(error['error_equation'].strip())
                        
                    #if key=='single_task_wor':
                        #print(response_equa_list)
                        #print(ground_truth_teacher)
                        #print('=========================')
                    all_fit=1
                    for ground_truth in ground_truth_teacher:
                        flag=0
                        for equa in response_equa_list:
                            if equa==ground_truth:
                                flag=1
                            else:
                                break
                        if flag==0:
                            all_fit=0
                            break

                    if all_fit==1:
                        correct_dic[key][1]+=1
                    correct_dic[key][2]+=calculate_distance(dic,response_equa_list)

                else:
                    correct_dic[key][2]+=DISTANCE_PENALTY

for key, value in correct_dic.items():
    print(f'{key}:')
    print(f'EM : {value[1]/value[0] *100:20.2f}%')
    print(f'DIS : {value[2]/value[0] :20.2f}')
    print()
        


print('leave one out\n')


sum_dic={}
correct_dic={}
for key, value in responses[0].items():
    if 'wr' not in key and 'wor' not in key:
        continue
    else:
        # [ground turth wrong amount, em count, dis sum]
        correct_dic[key]=[0,0,0]
        # [em, dis]
        sum_dic[key]=[0,0]

sum_dic={}

for n in range(1,7):
    with open(f'model_response/leave_one_out_student{n}_problem-solving_error_identification.json', 'r') as f:
        responses=json.load(f)
        
    
    correct_dic={}
    for key, value in responses[0].items():
        if 'wr' not in key and 'wor' not in key:
            continue
        else:
            # [ground turth wrong amount, em count, dis sum]
            correct_dic[key]=[0,0,0]
            # [em, dis]
            if key not in sum_dic:
                sum_dic[key]=[0,0]
    for dic in responses:
        for key, value in dic.items():
            if 'wr' not in key and 'wor' not in key:
                continue
            else:
                if dic['correct_or_not']=='wrong':
                    correct_dic[key][0]+=1
                    if value != '' and value != "The student's answer doesn't contain wrong equations.":
                        if key == 'o1_mini_wor':
                            response_equa_list=[value]
                        else:
                            response_equa_list = get_wrong_equation(value)
                        if len(response_equa_list)==0:
                            correct_dic[key][2]+=DISTANCE_PENALTY
                            continue
                        ground_truth_teacher=[]
                        for error in dic['teacher_review']['error']:
                            ground_truth_teacher.append(error['error_equation'].strip())

                        #if key=='single_task_wor':
                            #print(response_equa_list)
                            #print(ground_truth_teacher)
                            #print('=========================')
                        all_fit=1
                        for ground_truth in ground_truth_teacher:
                            flag=0
                            for equa in response_equa_list:
                                if equa==ground_truth:
                                    flag=1
                                else:
                                    break
                            if flag==0:
                                all_fit=0
                                break

                        if all_fit==1:
                            correct_dic[key][1]+=1
                        correct_dic[key][2]+=calculate_distance(dic,response_equa_list)

                    else:
                        correct_dic[key][2]+=DISTANCE_PENALTY

    for key, value in correct_dic.items():
        sum_dic[key][0]+=value[1]/value[0] *100
        sum_dic[key][1]+=value[2]/value[0]
        
for key, value in sum_dic.items():   
    print(f'{key}:')
    print(f'EM : {value[0]/6:20.2f}%')
    print(f'DIS : {value[1]/6:20.2f}')
    print()



print('Feedback Generation')
print('time series split\n')

with open('../model response/time_series_split_feedback_generation.json', 'r') as f:
    responses=json.load(f)

score_tmp=[{'rouge-1':{'r':0,'p':0,'f':0},'rouge-2':{'r':0,'p':0,'f':0},'rouge-l':{'r':0,'p':0,'f':0}},0,[],[]]
score_dic={}
for key, value in responses[0].items():
    if 'wr' not in key and 'wor' not in key:
        continue
    else:
        # [rouge1, rouge2, rougel, error counts, condidate list, references list]
        score_dic[key]=copy.deepcopy(score_tmp)

for dic in responses:
        for key, value in dic.items():
            if 'wor' not in key and 'wr' not in key:
                continue
            else:           
                if dic['correct_or_not']=='wrong':
                    score_dic[key][1]+=1
                    if value.strip()!='' and value!="The student's answer is correct and doesn't need advice." :
                        feed_back = get_teacher_advice(value)
                        if feed_back!='':
                            ground_truth_teacher=''
                            for error in dic['teacher_review']['error']:
                                ground_truth_teacher+=error['teacher_advice_en']

                            #print(key)
                            #print(value)
                            scores = rouge.get_scores(get_teacher_advice(value), ground_truth_teacher) 
                            score_dic[key][0]['rouge-1']['r']+=scores[0]['rouge-1']['r']
                            score_dic[key][0]['rouge-1']['p']+=scores[0]['rouge-1']['p']
                            score_dic[key][0]['rouge-1']['f']+=scores[0]['rouge-1']['f']
                            score_dic[key][0]['rouge-2']['r']+=scores[0]['rouge-2']['r']
                            score_dic[key][0]['rouge-2']['p']+=scores[0]['rouge-2']['p']
                            score_dic[key][0]['rouge-2']['f']+=scores[0]['rouge-2']['f']
                            score_dic[key][0]['rouge-l']['r']+=scores[0]['rouge-l']['r']
                            score_dic[key][0]['rouge-l']['p']+=scores[0]['rouge-l']['p']
                            score_dic[key][0]['rouge-l']['f']+=scores[0]['rouge-l']['f']

                            score_dic[key][2].append(get_teacher_advice(value))
                            score_dic[key][3].append(ground_truth_teacher)

                        
for key, value in score_dic.items():
    print(f'{key} :')   
    P, R, F1 = score(score_dic[key][2], score_dic[key][3], lang="en", verbose=True)
    print(f"teacher advice bert score : {F1.mean()*len(score_dic[key][2])/score_dic[key][1]:.4f}")
    print(f"teacher advice rouge-1 f1: {score_dic[key][0]['rouge-1']['f']/score_dic[key][1]:.4f}, rouge-2 f1: {score_dic[key][0]['rouge-2']['f']/score_dic[key][1]:.4f}, rouge-l f1: {score_dic[key][0]['rouge-l']['f']/score_dic[key][1]:.4f}")


print('leave one out\n')

sum_dic={}

for n in range(1,7):
    with open(f'model_response/leave_one_out_student{n}_feedback_generation.json', 'r') as f:
        responses=json.load(f)
    
    score_tmp=[{'rouge-1':{'r':0,'p':0,'f':0},'rouge-2':{'r':0,'p':0,'f':0},'rouge-l':{'r':0,'p':0,'f':0}},0,[],[]]
    score_dic={}
    for key, value in responses[0].items():
        if 'wr' not in key and 'wor' not in key:
            continue
        else:
            # [rouge1, rouge2, rougel, error counts, condidate list, references list]
            score_dic[key]=copy.deepcopy(score_tmp)
            if key not in sum_dic:
                #rouge1, rouge2, rougel, bertscore]
                sum_dic[key] = [0,0,0,0]

    for dic in responses:
        for key, value in dic.items():
            if 'wor' not in key and 'wr' not in key:
                continue
            else:           
                if dic['correct_or_not']=='wrong':
                    score_dic[key][1]+=1
                    if value.strip()!='' and value!="The student's answer is correct and doesn't need advice." :
                        feed_back = get_teacher_advice(value)
                        if feed_back!='':
                            ground_truth_teacher=''
                            for error in dic['teacher_review']['error']:
                                ground_truth_teacher+=error['teacher_advice_en']

                            #print(key)
                            #print(value)
                            scores = rouge.get_scores(get_teacher_advice(value), ground_truth_teacher) 
                            score_dic[key][0]['rouge-1']['r']+=scores[0]['rouge-1']['r']
                            score_dic[key][0]['rouge-1']['p']+=scores[0]['rouge-1']['p']
                            score_dic[key][0]['rouge-1']['f']+=scores[0]['rouge-1']['f']
                            score_dic[key][0]['rouge-2']['r']+=scores[0]['rouge-2']['r']
                            score_dic[key][0]['rouge-2']['p']+=scores[0]['rouge-2']['p']
                            score_dic[key][0]['rouge-2']['f']+=scores[0]['rouge-2']['f']
                            score_dic[key][0]['rouge-l']['r']+=scores[0]['rouge-l']['r']
                            score_dic[key][0]['rouge-l']['p']+=scores[0]['rouge-l']['p']
                            score_dic[key][0]['rouge-l']['f']+=scores[0]['rouge-l']['f']

                            score_dic[key][2].append(get_teacher_advice(value))
                            score_dic[key][3].append(ground_truth_teacher)


    for key, value in score_dic.items():
        #print(f'{key} :')   
        #print()
        if len(score_dic[key][2])!=0:
            P, R, F1 = score(score_dic[key][2], score_dic[key][3], lang="en", verbose=True)
            #print(f"teacher advice bert score : {F1.mean()*len(score_dic[key][2])/score_dic[key][1]:.4f}")
            #print(f"teacher advice rouge-1 f1: {score_dic[key][0]['rouge-1']['f']/score_dic[key][1]:.4f}, rouge-2 f1: {score_dic[key][0]['rouge-2']['f']/score_dic[key][1]:.4f}, rouge-l f1: {score_dic[key][0]['rouge-l']['f']/score_dic[key][1]:.4f}")
            sum_dic[key][0]+=score_dic[key][0]['rouge-1']['f']/score_dic[key][1]
            sum_dic[key][1]+=score_dic[key][0]['rouge-2']['f']/score_dic[key][1]
            sum_dic[key][2]+=score_dic[key][0]['rouge-l']['f']/score_dic[key][1]
            sum_dic[key][3]+=F1.mean()*len(score_dic[key][2])/score_dic[key][1]
for key, value in sum_dic.items():
    print(key)
    print(f'rouge-1 {sum_dic[key][0]/6:.4f}')
    print(f'rouge-2 {sum_dic[key][1]/6:.4f}')
    print(f'rouge-l {sum_dic[key][2]/6:.4f}')
    print(f'bert score  {sum_dic[key][3]/6:.4f}')
    print()



for n in range(6): 
    with open(fr'C:\Users\potot\pythonCode\NLP\Paper_experiment\Dataset_process\dataset split\type1_train_student_{n}.json', 'r') as f:
        train_data=json.load(f)
    random.seed(42)
    random.shuffle(train_data)

    with open(fr'C:\Users\potot\pythonCode\NLP\Paper_experiment\Dataset_process\dataset split\type1_test_student_{n}.json', 'r') as f:
        test_data=json.load(f)
        
    examples=[]

    count=0
    for element in train_data:
        if element['correct_or_not']=='correct':
            examples.append(element)
            count+=1
        if count==3:
            break

    for element in train_data:
        if element['correct_or_not']=='wrong':
            if element['teacher_review']['error'][0]['error_equation']=='None':
                examples.append(element)
                break

    for element in train_data:
        if element['correct_or_not']=='wrong':
            if element['teacher_review']['error'][0]['error_equation']!='None':
                if element['teacher_review']['error_counts']==1:
                    examples.append(element)
                    break

    for element in train_data:
        if element['correct_or_not']=='wrong':
            if element['teacher_review']['error'][0]['error_equation']!='None':
                if element['teacher_review']['error_counts']>1:
                    examples.append(element)
                    break

    random.seed(42)
    random.shuffle(examples)
    #print(test_data[0]['student_id'])
    #print('=============================')
    
    for element in tqdm(test_data):
        if 'llama3_8b_checking_response' in element.keys():
            continue
        prompt=create_prompt("woR",examples,element)
        chat_completion = client.chat.completions.create(
            messages=prompt,
            model="llama3-8b-8192",
        )
        element['llama3_8b_checking_response'] = chat_completion.choices[0].message.content

        time.sleep(2)
        
    with open(f'llama3_8b_checking_result_type1_stu{n}_woR.json', 'w') as f:
        json.dump(test_data, f)

