# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 13:17:54 2018

@author: Jack
"""
def get_Flexion(a):
    flexion = 1
    if len(a.split(";"))>1:
        b = a.split(";")[0]
    else:
        b = a
    if "[" in b:
        flexion+=1
    if "<" in b:
        flexion+=2
    if "(" in b:
        flexion+=3
    if "C" in b:
        flexion+=4
    if "O" in b:
        flexion+=5
    if ">" in b:
        flexion+=5
    if "@" in b:
        flexion+=6
    return flexion

def get_thumb_flexion(a):
    flex = get_Flexion(a)
    if (a.find('T') < a.find(';')) and ("-" not in a):
        thumb_flex = 1
    elif (a.find('T') > a.find(';')) and (flex<3)  and ("-" not in a):
        thumb_flex = 1
    elif ("T-" in a) and (flex==7):
        thumb_flex = 1
    elif "T-" in a:
        thumb_flex = -1
    else:
        thumb_flex = 0
    return thumb_flex

def get_NSFFlexion(a):
    #NSF flexion
    nsf_flexion=0
    if "/" in a: #extended
        nsf_flexion-=1
    if "#" in a: #closed
        nsf_flexion+=1
    return nsf_flexion

def get_nsfflexion_thumb(a):
    nsf_flexion = get_NSFFlexion(a)
    if "T-" in a:
        thumb = -1
    elif "T" in a:
        thumb = 1
    else:
        thumb = 0
    return thumb+nsf_flexion


def get_FingComplexity(a):
    compF = 0
    b = a.split(" ")
    if len(b)>1:
        c = [x[0] for x in b]
        if len(set(c)) > 1:
            compF+=1
        #if a.split(" ")[0][0] !=  a.split(" ")[1][0]:
            #compF+=1
        a = b[0]
    if a.count(";") == 1:
        compF+= 2
    elif a.count(";") > 1:
        compF+= 3
    elif a[0] in  ["T","B","1"]:
        compF+= 1
    elif a[0] in ["J","U","8"]:
        compF+= 2
    elif a[0] in ["M","P","D","H","A","2","7"]:
        compF+= 3
    else:
        compF = "ERROR"
    #compF = compF/len(a.split(" "))
    return compF

def get_JointComplexity(a):
    compJ = 0
    if "K" in a:
        compJ+= 3
    elif "X" in a:
        compJ+= 3
    elif "<" in a:
        compJ+= 2
    elif ">" in a:
        compJ+= 2
    elif  "C" in a:
        compJ+= 2
    elif  "O" in a:
        compJ+= 2
    elif "(" in a:
        compJ+= 2
    elif "[" in a:
        compJ+= 2
    elif "^" in a:
        compJ+= 2
    else:
        compJ+= 1
    #compJ = compJ/len(a.split(" "))
    return compJ

def get_SelectedFing(a):
    selfing = ""
    if "M" in a:
        selfing+="imr"
    if "P" in a:
        selfing+="mp"
    if "B" in a:
        selfing+="imrp"
    if "D" in a:
        selfing+="mrp"
    if "U" in a:
        selfing+="im"
    if "H" in a:
        selfing+="ip"
    if "A" in a:
        selfing+="mr"
    if "2" in a:
        selfing+="rp"
    if "1" in a:
        selfing+="i"
    if "8" in a:
        selfing+="m"
    if "7" in a:
        selfing+="r"
    if "J" in a:
        selfing+="p"
    if "T" in a:
        selfing+="thumb"
    return selfing    

def get_apChange(a):
    b = a.split(" ")
    apChange = 0
    if len(b) == 1:
        return apChange
    else:
        c = [get_Flexion(x) for x in b]
        for i in range(len(c)-1):
            diff = abs(c[i+1] - c[i])
            if diff > 3:
                apChange +=1
                break
        #print(c)
        if (apChange==0) and (abs(c[-1:][0] - c[0]) > 3):
            apChange +=1
    return apChange

def featureCoding(a):
    a = a.upper()
    b = a.split(" ")[0]
    aperture_change = 0
    extra_point = 0
    if len(a.split(" "))>1:
        #flexion = []
        jointComp = []
        for x in a.split(" "):
            #flexion.append(get_Flexion(x))
            if "K" in x:
                jointComp.append(1)
            elif "X" in x:
                jointComp.append(1)
            else:
                jointComp.append(0)
        #for i in range(len(flexion)-1):
            #diff = abs(flexion[i+1] - flexion[i])
            #if diff > 3:
                #aperture_change+=1
                #break
        if sum(jointComp) != len(a.split(" ")):
            if sum(jointComp) != 0:
                extra_point+=1
        

    compF = get_FingComplexity(a)
    compJ = get_JointComplexity(b) + aperture_change + extra_point
    flexion = get_Flexion(b)
    nsf_flexion = get_NSFFlexion(b)
    selfing = get_SelectedFing(b)
    aperture_change = get_apChange(a)
    nsf_thumb = get_nsfflexion_thumb(b) 
    thumb_flex = get_thumb_flexion(b) 

    return compF,compJ,flexion,nsf_flexion,selfing,aperture_change,nsf_thumb,thumb_flex
