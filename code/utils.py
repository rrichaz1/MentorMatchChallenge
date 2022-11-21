from sentence_transformers import SentenceTransformer, util
import numpy as np
import re
import pandas as pd

timezonemap ={'IST': 0,'CET': 2,'EST': 3, 'CST': 4, 'MST': 5, 'PST':6}
model = SentenceTransformer('stsb-roberta-large')

def intersection(lst1, lst2):
 
    #https://www.geeksforgeeks.org/python-intersection-two-lists/
    #https://bobbyhadz.com/blog/python-split-string-ignore-empty-strings
    # Use of hybrid method 
    temp = set([value for value in lst2 if value])
    lst3 = [value for value in lst1 if value and value in temp]
    return lst3

def validate_mentormentee(mentor_row, mentee_row, result):   
    check_mentorgrade_eligibility(mentor_row, mentee_row, result)
    check_mentordepartment_match(mentor_row, mentee_row, result)
    check_goal_similarity(mentor_row, mentee_row, result)
    assignTimeZoneNumericValue(mentor_row, mentee_row, result)
    matchInterests(mentor_row, mentee_row, result)
    matchCLLGrowth(mentor_row, mentee_row, result)
    checkCircleLevel(mentor_row, mentee_row, result)
    return result    

def check_mentorgrade_eligibility(mentor_row, mentee_row,  result):   
    
        
    if mentor_row['Mentor_GradeLevel'].dtype == np.int64:
        result['Mentor_Grade'] = mentor_row['Mentor_GradeLevel'].iloc[0]
    else:
        result['Mentor_Grade'] = int(re.search(r'[0-9]+', mentor_row['Mentor_GradeLevel'].iloc[0]).group())
        
        
    if mentee_row['Mentee_GradeLevel'].dtype == np.int64:
        result['Mentee_Grade'] = mentee_row['Mentee_GradeLevel'].iloc[0]
    else:
        result['Mentee_Grade'] = int(re.search(r'[0-9]+', mentee_row['Mentee_GradeLevel'].iloc[0]).group())
        
    if result['Mentor_Grade'] <= 28 :
        result['Mentor_Eligible_Error'] = 1
    else:
        result['Mentor_Eligible_Error'] = 0
        
    if result['Mentor_Grade'] <= result['Mentee_Grade'] :
        result['MentorMentee_Grade_Error'] = 1
    else:
        result['MentorMentee_Grade_Error'] = 0    
    
    return result  

def check_mentordepartment_match(mentor_row, mentee_row,  result):   
    
        
    result['Mentor_Department'] = mentor_row['Mentor_Department'].iloc[0]
    result['Mentee_Department'] = mentee_row['Mentee_Department'].iloc[0]
   
        
    if result['Mentor_Department'] == result['Mentee_Department'] :
        result['MentorMentee_DeptError'] = 1
    else:
        result['MentorMentee_DeptError'] = 0    
    
    return result  

def get_similarity(mentor_goal, mentee_goal):
    
    embedding1 = model.encode(mentor_goal, convert_to_tensor=True)
    embedding2 = model.encode(mentee_goal, convert_to_tensor=True)
    # compute similarity scores of two embeddings
    cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
   
    return round(cosine_scores.item(), 4)

def check_goal_similarity(mentor_row, mentee_row, result):
   
    mentor_goal= mentor_row['Mentor_Goal'].iloc[0].replace('"','')
    mentee_goal= mentee_row['Mentee_Goal'].iloc[0].replace('"','')
   
    #print("Similarity score:", cosine_scores.item())
    result['Goal_Similarity'] =get_similarity(mentor_goal, mentee_goal)

    
    return result



def assignTimeZoneNumericValue(mentor_row, mentee_row, result):
    mentorTimeZone = re.search(r'\(([^)]+)', mentor_row['Mentor_TimeZone'].iloc[0]).group(1)
    menteeTimeZone = re.search(r'\(([^)]+)', mentee_row['Mentee_TimeZone'].iloc[0]).group(1)
    mentorTimeZoneNum = timezonemap.get(mentorTimeZone)
    menteeTimeZoneNum = timezonemap.get(menteeTimeZone)
    result['Mentor_TimeZone'] = mentorTimeZone
    result['Mentee_TimeZone'] = menteeTimeZone
    result['Mentor_TimeZoneNum'] = mentorTimeZoneNum
    result['Mentee_TimeZoneNum'] = menteeTimeZoneNum
    result['Mentor_Mentee_TimeZoneDist'] = abs(menteeTimeZoneNum - mentorTimeZoneNum)
    return result

def matchInterests(mentor_row, mentee_row, result):
    mentor_interests = mentor_row['Mentor_Interest'].iloc[0].replace('"','').split(';')
    mentee_interests = mentee_row['Mentee_Interest'].iloc[0].replace('"','').split(';')
    matchingInterests = len(intersection(mentor_interests, mentee_interests))
    
            
    result['MentorMentee_Common_Interests'] = matchingInterests
    return result

def matchCLLGrowth(mentor_row, mentee_row, result):
    mentor_clls = mentor_row['Mentor_CLL'].iloc[0].replace('"','').split(';')
    mentee_clls = mentee_row['Mentee_CLL'].iloc[0].replace('"','').split(';')
    mentee_cll3 = [menteecll for menteecll in mentee_clls if menteecll] [-4:-1]
    mentor_cll3 = [mentorcll for mentorcll in mentor_clls if mentorcll] [0:3]
    matchingCLL = len(intersection(mentee_cll3, mentor_cll3))
    
            
    result['MentorMentee_CLL_Growth'] = matchingCLL
    return result


def checkCircleLevel(mentor_row, mentee_row, result):
    mentor_circle = mentor_row['Mentor_Circle'].iloc[0].split('(')[0].strip()
    mentee_circle = mentee_row['Mentee_Circle'].iloc[0].split('(')[0].strip()
    if mentor_circle != mentee_circle:
        result['MentorMentee_Circle_Error'] = 1
    else:
        result['MentorMentee_Circle_Error'] = 0
    
    result['Mentor_Circle'] = mentor_circle  
    result['Mentee_Circle'] = mentee_circle  
    
    return result


def calculateCircleLevelScore(regroup_DF, Mentee_Department_In_Circle_Error_Group):
    regroup_dict = regroup_DF.to_dict('records')
    Circle_Summary_Dict = {}
    
    Circle_Summary_DF = pd.DataFrame()

    score = 0
    for row in regroup_dict:
        circle = row[('Circle', '')]
        Circle_Summary_Dict['Circle'] = circle
        penalty = calculateCircleLevelPenaltyScore(row, Circle_Summary_Dict, circle, Mentee_Department_In_Circle_Error_Group)
        bonus = calculateCircleLevelBonusScore(row, Circle_Summary_Dict)
        score += (penalty+bonus)
        Circle_Summary_DF = Circle_Summary_DF.append(Circle_Summary_Dict, ignore_index=True)
    
    return (score, Circle_Summary_DF)

def calculateCircleLevelPenaltyScore(row, Circle_Summary_Dict, circle, Mentee_Department_In_Circle_Error_Group):    
    # 2 mentors cannot be assigned to 
    score = 0
    mentorUnique = row[('Mentor_ID', 'nunique')]
    Circle_Summary_Dict['Unique Mentors']  = mentorUnique
    Circle_Summary_Dict['Unique Mentors Penalty'] = 0
    if mentorUnique > 1:
        Circle_Summary_Dict['Unique Mentors Penalty'] = -2
        score -= 2


    menteeUnique = row[('Mentee_ID', 'nunique')]
    Circle_Summary_Dict['Circle Size']  = menteeUnique
    Circle_Summary_Dict['Circle Size Penalty'] = 0
    if menteeUnique > 10:
        Circle_Summary_Dict['Circle Size Penalty'] = -1
        score -= 1
    

    # Mentees spread across time zones - makes it difficult to manage
    menteeTimeZoneDiff = row[('Mentee_TimeZoneNum', 'max')]- row[('Mentee_TimeZoneNum', 'min')]
    Circle_Summary_Dict['Mentees TimeZone Difference'] = menteeTimeZoneDiff
    Circle_Summary_Dict['Mentees TimeZone Penalty'] = 0
    if menteeTimeZoneDiff > 3:
        Circle_Summary_Dict['Mentees TimeZone Penalty'] = -1
        score -= 1

    # Mentors spread across time zones - makes it difficult to manage    
    mentorMenteeTimeZoneDiff = row[('Mentor_Mentee_TimeZoneDist', 'max')]    
    Circle_Summary_Dict['Mentee Mentor TimeZone Difference'] = mentorMenteeTimeZoneDiff
    Circle_Summary_Dict['Mentee Mentor TimeZone Penalty'] = 0
    if mentorMenteeTimeZoneDiff > 3:
        Circle_Summary_Dict['Mentee Mentor TimeZone Penalty'] = -1
        score -= 1

    #Mentor not eligible - Grade level < 28
    mentorEligibleError = 1 if row[('Mentor_Eligible_Error', 'sum')]  else 0
    Circle_Summary_Dict['Mentor Eligibility Penalty'] = 0
    Circle_Summary_Dict['Mentor Eligible Error'] =mentorEligibleError
    if mentorEligibleError:
        Circle_Summary_Dict['Mentor Eligibility Penalty'] = -1
        score -= 1

    #Mentor Mentee should not be from same department
    mentorMenteeDeptError = 1 if row[('MentorMentee_DeptError', 'sum')] else 0
    Circle_Summary_Dict['Mentee Mentor Department Penalty'] = 0
    Circle_Summary_Dict['Mentee Mentor Department Error'] =mentorMenteeDeptError
    if mentorMenteeDeptError:
        Circle_Summary_Dict['Mentee Mentor Department Penalty'] = -1
        score -= 1


    #More than 2 Mentees should not be from same department in the same circle   
    menteeDeptUnique = row[('Mentee_Department', 'nunique')]
    Circle_Summary_Dict['Unique Mentee Departments']  = menteeDeptUnique
    menteeDeptError = 1 if menteeUnique - menteeDeptUnique else 0
    Circle_Summary_Dict['Mentee Mentor Department Penalty'] = 0
    Circle_Summary_Dict['Mentee Mentor Department Error'] =mentorMenteeDeptError
    if mentorMenteeDeptError:
        Circle_Summary_Dict['Mentee Mentor Department Penalty'] = -1
        score -= 1

    #Check if there are too many mentees belonging to same department
    Circle_Summary_Dict['Too Many Mentees in Same Department Penalty'] = 0
    if circle in Mentee_Department_In_Circle_Error_Group:
        Circle_Summary_Dict['Too Many Mentees in Same Department Penalty'] = -1
        score -= 1


    #If mentor and mentees chose different circle
    menteeMentorCircleError = row[('MentorMentee_Circle_Error', 'sum')]
    Circle_Summary_Dict['Mentor Mentee Circle Error'] = menteeMentorCircleError
    Circle_Summary_Dict['Mentor Mentee Circle Penalty'] = 0
    if menteeMentorCircleError:
        Circle_Summary_Dict['Mentor Mentee Circle Penalty'] = -1
        score -= 1
        
    return score    

def calculateCircleLevelBonusScore(row, Circle_Summary_Dict):    
    ##Calculate bonus
    score = 0
    #Minimun common mentee growth and mentor strength
    menteeMentorCLLGrowth = row[('MentorMentee_CLL_Growth', 'min')]
    Circle_Summary_Dict['Mentor Mentee CLL Growth Commonality'] = menteeMentorCLLGrowth
    Circle_Summary_Dict['Mentor Mentee CLL Growth Bonus'] = 0
    if menteeMentorCLLGrowth > 1:
        Circle_Summary_Dict['Mentor Mentee CLL Growth Bonus'] = 1
        score += 1


    #More common interests gets bonus - check for minimum
    menteeMentorCommonInterestsMin = row[('MentorMentee_Common_Interests', 'min')]
    menteeMentorCommonInterestsMax = row[('MentorMentee_Common_Interests', 'min')]
    Circle_Summary_Dict['Mentor Mentee Minimum Interest Commonality'] = str(menteeMentorCommonInterestsMin) +" - "+ str(menteeMentorCommonInterestsMax)
    Circle_Summary_Dict['Mentor Mentee Interest Bonus'] = 0
    if menteeMentorCommonInterestsMin > 2:
        Circle_Summary_Dict['Mentor Mentee Interest Bonus'] = 2
        score += 2


    #Closer range of common interests gets bonus
    menteeMentorCommonGoalsMin = row[('Goal_Similarity', 'min')]
    menteeMentorCommonGoalsMax = row[('Goal_Similarity', 'max')]
    Circle_Summary_Dict['Mentor Mentee Min Goal Similarity'] = str(menteeMentorCommonGoalsMin) +" - "+ str(menteeMentorCommonGoalsMax)
    Circle_Summary_Dict['Mentor Mentee Goal Similarity Bonus'] = 0
    if menteeMentorCommonGoalsMin > 0.2 and (menteeMentorCommonInterestsMax - menteeMentorCommonInterestsMin) < 0.3:
        Circle_Summary_Dict['Mentor Mentee Goal Similarity Bonus'] = 2
        score += 2

    return score    
        
def createSummary(Result_DF, regroup_DF):
    Summary_Results = {}
    score = 0
   
    # Same mentee assigned more than once
    Summary_Results['Mentee Assigned More Than Once'] =";".join(set(Result_DF.groupby('Mentee_Email').filter(lambda x: len(x) >= 2).Mentee_Email))
    # Circles are uneven in size
    Summary_Results['Uneven Circle Size'] = 'No'
    maxCircleSize = regroup_DF.iloc[:,3].max()
    minCircleSize = regroup_DF.iloc[:,3].min()
    Summary_Results['Span Circle Size'] = str(minCircleSize)+" - "+str(maxCircleSize)
    if (maxCircleSize - minCircleSize) >5:
        Summary_Results['Uneven Circle Size Penalty'] = -2
        score -= 2
    
    #Find if same mentor is assigned to 2 circles.
    #Group by Circle and Mentor - on resulting object group by Mentor Id - select balue if the row appears twice
    #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.groupby.GroupBy.size.html
    
    sub = pd.DataFrame(Result_DF[['Circle', 'Mentor_ID']].value_counts())
    sub = sub.reset_index()
    MentorIn2CirclesDF = sub.groupby('Mentor_ID', as_index=False).size()

    MentorIn2CirclesList = MentorIn2CirclesDF[MentorIn2CirclesDF['size']>1]['Mentor_ID'].tolist()
    if len(MentorIn2CirclesList) > 0:
        Summary_Results['Mentor assigned to 2 circles penalty'] = -2
        Summary_Results['Mentor assigned to 2 circles '] = ";".join( [str(x) for x in MentorIn2CirclesList] )
        score -=5
        
    return   (score, Summary_Results)
    