# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 19:30:37 2018

@author: agabh
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 17:43:53 2018

@author: agabh
"""

import re
from io import open
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn import svm
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import 

#nv3='C:\\Users\\agabh\\Desktop\\NLP\\project\\data\\nv3'
##nv5='C:\\Users\\agabh\\Desktop\\NLP\\project\\data\\nv5'
nv3='C:\\Users\\agabh\\Desktop\\NLP\\project\\data\\nv6'
#nv8='C:\\Users\\agabh\\Desktop\\NLP\\project\\data\\nv8'
#v3='C:\\Users\\agabh\\Desktop\\NLP\\project\\data\\v3'
#v5='C:\\Users\\agabh\\Desktop\\NLP\\project\\data\\v5'
v3='C:\\Users\\agabh\\Desktop\\NLP\\project\\data\\v6'
#v8='C:\\Users\\agabh\\Desktop\\NLP\\project\\data\\v8'
#f=open('coeff.txt','r')
#coeff=f.read()

#f.close
#f=open('vocab.txt','r')
#vocab=f.read()
def corpus(address,rr):
    list_doc=os.listdir(address)
    corp=[]
    for i in range(0,len(list_doc)):
        
        strr=address+'\\'+list_doc[i]
#        print strr
        with open("%s"% strr,'r', encoding='utf8') as f:
            doc=f.read()
        if rr==1:
            doc1 =extractFacts1(doc)
        if rr==2:
            doc1 =extractFacts2(doc)
        if rr==3:
            doc1 =extractFacts3(doc)
        corp.append(doc1)
    return corp
        
def out_arr(len1,len2,j):
    mat=np.zeros((len2,len1))
    for i in range(0,len2):
        mat[i][j]=1
    return mat


def extractFacts2(data):
    pat=re.compile(r'\nTHE FACTS\n+')
    pat2=re.compile(r'\nTHE LAW\n+')
#    pat3=re.compile(r'\n+')
    i=re.finditer(pat,data)
    j=re.finditer(pat2,data)
    ind=-1
    ind1=-1
    for mat in i:
        ii=mat.end()
        ind=ii
        break
    
    for mat in j:
        jj=mat.start()
        ind1=jj
        break
    str1=data[ind:ind1]
    p = re.compile(r'[0-9]+\.?')
    str1 = re.sub(p,"",str1)
    p1=re.compile(r'(mrs|pp|la|eu|mr|et|ft|dr|azerbaijan|russian)')
#    mat=p1.findall(str1)
#    print mat
    
#    str1 = re.sub(p1,"",str1)
#    if ind==-1:
#        return -1
#    else:
    return str1

def extractFacts1(data):
    pat=re.compile(r'\nPROCEDURE\n+')
    pat2=re.compile(r'\nTHE FACTS\n+')
#    pat3=re.compile(r'\n+')
    i=re.finditer(pat,data)
    j=re.finditer(pat2,data)
    ind=-1
    ind1=-1
    for mat in i:
        ii=mat.end()
        ind=ii
        break
    
    for mat in j:
        jj=mat.start()
        ind1=jj
        break
    str1=data[ind:ind1]
    p = re.compile(r'[0-9]+\.?')
    str1 = re.sub(p,"",str1)
    p1=re.compile(r'(mrs|pp|la|eu|mr|et|ft|dr|azerbaijan|russian)')
#    mat=p1.findall(str1)
#    print mat
    
#    str1 = re.sub(p1,"",str1)
#    if ind==-1:
#        return -1
#    else:
    return str1

def extractFacts3(data):
    pat=re.compile(r'\nTHE LAW\n+')
    pat2=re.compile(r'\nFOR THESE\n+')
#    pat3=re.compile(r'\n+')
    i=re.finditer(pat,data)
    j=re.finditer(pat2,data)
    ind=-1
    ind1=-1
    for mat in i:
        ii=mat.end()
        ind=ii
        break
    
    for mat in j:
        jj=mat.start()
        ind1=jj
        break
    str1=data[ind:ind1]
    p = re.compile(r'[0-9]+\.?')
    str1 = re.sub(p,"",str1)
    p1=re.compile(r'(mrs|pp|la|eu|mr|et|ft|dr|azerbaijan|russian)')
#    mat=p1.findall(str1)
#    print mat
    
#    str1 = re.sub(p1,"",str1)
#    if ind==-1:
#        return -1
#    else:
    return str1

corpus_nv3_1=corpus(nv3,1)
corpus_nv3_2=corpus(nv3,2)
corpus_nv3_3=corpus(nv3,3)
#corpus_nv5=corpus(nv5)
#corpus_nv6=corpus(nv6)
#corpus_nv8=corpus(nv8)
corpus_v3_1=corpus(v3,1)
corpus_v3_2=corpus(v3,2)
corpus_v3_3=corpus(v3,3)
#corpus_v5=corpus(v5)
#corpus_v6=corpus(v6)
#corpus_v8=corpus(v8)

corpus_f_1=corpus_nv3_1+corpus_v3_1
corpus_f_2=corpus_nv3_2+corpus_v3_2
corpus_f_3=corpus_nv3_3+corpus_v3_3
tf_vect=TfidfVectorizer(lowercase=True,stop_words='english',min_df=50,max_df=0.5,encoding='ascii')

tf_1=tf_vect.fit_transform(corpus_f_1).toarray()
ff_1=tf_vect.get_feature_names()

tf_2=tf_vect.fit_transform(corpus_f_2).toarray()
ff_2=tf_vect.get_feature_names()

tf_3=tf_vect.fit_transform(corpus_f_3).toarray()
ff_3=tf_vect.get_feature_names()

y1_1=np.zeros(len(corpus_nv3_1))
y2_1=np.ones(len(corpus_v3_1))

y1_2=np.zeros(len(corpus_nv3_2))
y2_2=np.ones(len(corpus_v3_2))

y1_3=np.zeros(len(corpus_nv3_3))
y2_3=np.ones(len(corpus_v3_3))

yy_tr_1=np.concatenate((y1_1,y2_1))
yy_tr_2=np.concatenate((y1_2,y2_2))
yy_tr_3=np.concatenate((y1_3,y2_3))

#X_train, X_test, y_train, y_test = train_test_split(tf, yy_tr)
clf_1=svm.SVC(kernel='linear')
clf_1.fit(tf_1,yy_tr_1)
print "Fitting Done 1"
coeff_1=clf_1.coef_

clf_2=svm.SVC(kernel='linear')
clf_2.fit(tf_2,yy_tr_2)
print "Fitting Done 2"
coeff_2=clf_2.coef_

clf_3=svm.SVC(kernel='linear')
clf_3.fit(tf_3,yy_tr_3)
print "Fitting Done 3"
coeff_3=clf_3.coef_
#coeff=coeff.split()
coeff_1=coeff_1[0]
coeff_2=coeff_2[0]
coeff_3=coeff_3[0]

vocab_1=[]
for i in range(0,len(coeff_1)):
    voc=ff_1[i].encode('utf-8')
    vocab_1.append(voc)
    
vocab_2=[]
for i in range(0,len(coeff_2)):
    voc=ff_2[i].encode('utf-8')
    vocab_2.append(voc)
    
vocab_3=[]
for i in range(0,len(coeff_3)):
    voc=ff_3[i].encode('utf-8')
    vocab_3.append(voc)
    
indices1_1=np.argsort(coeff_1)[-35:]
indices2_1=np.argsort(coeff_1)[:35]
indices_1=np.concatenate((indices2_1,indices1_1))

indices1_2=np.argsort(coeff_2)[-35:]
indices2_2=np.argsort(coeff_2)[:35]
indices_2=np.concatenate((indices2_2,indices1_2))

indices1_3=np.argsort(coeff_3)[-35:]
indices2_3=np.argsort(coeff_3)[:35]
indices_3=np.concatenate((indices2_3,indices1_3))

feature_1=[]
weigh_t_1=[]
for i in range(0,len(indices_1)):
    ind=indices_1[i]
    feature_1.append(vocab_1[ind])
    weigh_t_1.append(coeff_1[ind])

feature_2=[]
weigh_t_2=[]
for i in range(0,len(indices_2)):
    ind=indices_2[i]
    feature_2.append(vocab_2[ind])
    weigh_t_2.append(coeff_2[ind])
    
feature_3=[]
weigh_t_3=[]
for i in range(0,len(indices_3)):
    ind=indices_3[i]
    feature_3.append(vocab_3[ind])
    weigh_t_3.append(coeff_3[ind])

corp_para_1=[]
for i in range(0,len(corpus_f_1)):
    patt=re.compile(r'(\n.+\n)')
#    patt2=re.compile(r'(\n)([0-9]+\.)')
    docc=re.findall(patt,corpus_f_1[i])
    for j in range(0,len(docc)):
        docc[j]=docc[j].encode('utf-8','ignore')
        docc[j]=docc[j].split()
#        docc[j]=set(docc[j])
#        docc[j]=list(docc[j])
    corp_para_1.append(docc)
print 'para generated 1'

corp_para_2=[]
for i in range(0,len(corpus_f_2)):
    patt=re.compile(r'(\n.+\n)')
#    patt2=re.compile(r'(\n)([0-9]+\.)')
    docc=re.findall(patt,corpus_f_2[i])
    for j in range(0,len(docc)):
        docc[j]=docc[j].encode('utf-8','ignore')
        docc[j]=docc[j].split()
#        docc[j]=set(docc[j])
#        docc[j]=list(docc[j])
    corp_para_2.append(docc)
print 'para generated 2'

corp_para_3=[]
for i in range(0,len(corpus_f_3)):
    patt=re.compile(r'(\n.+\n)')
#    patt2=re.compile(r'(\n)([0-9]+\.)')
    docc=re.findall(patt,corpus_f_3[i])
    for j in range(0,len(docc)):
        docc[j]=docc[j].encode('utf-8','ignore')
        docc[j]=docc[j].split()
#        docc[j]=set(docc[j])
#        docc[j]=list(docc[j])
    corp_para_3.append(docc)
print 'para generated 3'

def weight(l1,ind):
    ss=0
    for i in range(0,len(l1)):
        ss=ss+l1[i]
        if ind<=ss:
            return i
            
def score(l1,l2,w,l3):
    wei=0
    for i in range(0,len(l2)):
        strr=l2[i]

        try:
            ind=l1.index(strr)
#            ww=weight(l3,ind)
            www=w[ind]
    
        except ValueError:
            www=0
        
        wei+=www
    return wei

l81 = ['son', 'body', 'result', 'russian', 'department', 'prosecutor office', 'death', 'group', 'relative', 'head', 'described',
       'military', 'criminal investigation', 'burial', 'district prosecutor', 'men', 'deceased', 'town', 'attack', 'died']
l82 = ['health moral', 'law democratic', 'law democratic society', 'disorder crime', 'prevention disorder', 'prevention disorder'
       'crime', 'economic well', 'protection health', 'interest national', 'interest national security', 'public authority exercise',
       'interference public authority exercise', 'national security public', 'exercise law democratic', 'public authority exercise law',
       'authority exercise law democratic', 'exercise law', 'authority exercise law', 'exercise law democratic society', 'crime protection']
l83 = ['second', 'instance', 'second applicant', 'victim', 'municipal', 'violence', 'authorised', 'address', 'municipal court',
       'relevant provision', 'behaviour', 'register', 'appear', 'maintenance', 'instance court', 'defence', 'procedural', 'decide',
       'court decided', 'quashed']
l84 = ['service', 'obligation', 'data', 'duty', 'review', 'high', 'system', 'test', 'concern', 'building', 'agreed', 'professional',
       'positive', 'threat', 'carry', 'van', 'accepted', 'step', 'clear', 'panel']
l85 = ['contact', 'social', 'care', 'expert', 'opinion', 'living', 'welfare', 'county', 'physical', 'psychological', 'agreement', 'divorce',
       'restriction', 'support', 'live', 'dismissed applicant', 'prior', 'remained', 'court considered', 'expressed']
l86 = [' national', 'year', 'country', 'residence', 'minister', 'permit', 'requirement', 'netherlands', 'alien', 'board', 'claimed', 'stay',
       'contrary', 'objection', 'spouse', 'residence permit', 'close', 'deputy', 'deportation', 'brother']
l8 = []
l8_1=[]

l81_l=len(l81)
l82_l=len(l82)
l83_l=len(l83)
l84_l=len(l84)
l85_l=len(l85)
l86_l=len(l86)

ll_8=[l81_l,l82_l,l83_l,l84_l,l85_l,l86_l]

l8.append(l81)
l8.append(l82)
l8.append(l83)
l8.append(l84)
l8.append(l85)
l8.append(l86)

l8_1.extend(l81)
l8_1.extend(l82)
l8_1.extend(l83)
l8_1.extend(l84)
l8_1.extend(l85)
l8_1.extend(l86)
l8_w = [15.70, 12.20, 9.51, -7.89, -12.30, -13.50]

weight_f_1=[]
for i in range(0,len(corp_para_1)):
    ww=[]
    for j in range(0,len(corp_para_1[i])):
        l1=corp_para_1[i][j]
        
        weigh=score(feature_1,l1,weigh_t_1,ll_8)
        ww.append(weigh)
        
    weight_f_1.append(ww)
print 'scores distributed 1'

weight_f_2=[]
for i in range(0,len(corp_para_2)):
    ww=[]
    for j in range(0,len(corp_para_2[i])):
        l1=corp_para_2[i][j]
        
        weigh=score(feature_2,l1,weigh_t_2,ll_8)
        ww.append(weigh)
        
    weight_f_2.append(ww)
print 'scores distributed 2'

weight_f_3=[]
for i in range(0,len(corp_para_3)):
    ww=[]
    for j in range(0,len(corp_para_3[i])):
        l1=corp_para_3[i][j]
        
        weigh=score(feature_3,l1,weigh_t_3,ll_8)
        ww.append(weigh)
        
    weight_f_3.append(ww)
print 'scores distributed 3'
    
count1=0
count2=0
count3=0
cl=0
weight_f_1=np.array(weight_f_1)
weight_f_2=np.array(weight_f_2)
weight_f_3=np.array(weight_f_3)

out1_1=np.zeros(len(weight_f_1))
out1_2=np.zeros(len(weight_f_2))
out1_3=np.zeros(len(weight_f_3))

out2_1=np.zeros(len(weight_f_1))
out2_2=np.zeros(len(weight_f_2))
out2_3=np.zeros(len(weight_f_3))

for i in range(0,len(weight_f_1)):
    if len(weight_f_1[i])!=0:
        ma_x=np.max(weight_f_1[i])
        mi_n=np.min(weight_f_1[i])
        cl+=1
        if i<=len(corpus_nv3_1):
            count1=count1+(ma_x<=np.abs(mi_n))
            out1_1[i]=(ma_x>=np.abs(mi_n))
        else:
            count1=count1+(ma_x>=np.abs(mi_n))
            out1_1[i]=(ma_x>=np.abs(mi_n))
        
acc1_1=count1/float(len(out1_1))
print 'accuracy1_1=',acc1_1

for i in range(0,len(weight_f_2)):
    if len(weight_f_2[i])!=0:
        ma_x=np.max(weight_f_2[i])
        mi_n=np.min(weight_f_2[i])
        cl+=1
        if i<=len(corpus_nv3_2):
            count2=count2+(ma_x<=np.abs(mi_n))
            out1_2[i]=(ma_x>=np.abs(mi_n))
        else:
            count2=count2+(ma_x>=np.abs(mi_n))
            out1_2[i]=(ma_x>=np.abs(mi_n))
            
acc1_2=count2/float(len(out1_2))
print 'accuracy1_2=',acc1_2

for i in range(0,len(weight_f_3)):
    if len(weight_f_3[i])!=0:
        ma_x=np.max(weight_f_3[i])
        mi_n=np.min(weight_f_3[i])
        cl+=1
        if i<=len(corpus_nv3_3):
            count3=count3+(ma_x<=np.abs(mi_n))
            out1_3[i]=(ma_x>=np.abs(mi_n))
        else:
            count3=count3+(ma_x>=np.abs(mi_n))
            out1_3[i]=(ma_x>=np.abs(mi_n))

acc1_3=count3/float(len(out1_3))
print 'accuracy1_3=',acc1_3
#
#accuracy=count/float(cl)
#print accuracy

#count1=0
#for i in range(0,len(weight_f)):
#    if len(weight_f[i])!=0:
#        weight_f[i]=np.array(weight_f[i])
#        pos=np.sum(weight_f[i].clip(min=0))
#        neg=np.sum(weight_f[i].clip(max=0))
#        if i<=len(corpus_nv3):
#            count1=count1+(pos<=np.abs(neg))
#        else:
#            count1=count1+(pos>=np.abs(neg))
#
#accuracy1=count1/float(cl)
#print accuracy1
#
#count2=0
#for i in range(0,len(weight_f)):
#    if len(weight_f[i])!=0:
#        weight_f[i]=np.array(weight_f[i])
#        pos=np.sum(weight_f[i].clip(min=0))
#        neg=np.sum(weight_f[i].clip(max=0))
#        pos_num=np.count_nonzero(weight_f[i].clip(min=0))
#        neg_num=np.count_nonzero(weight_f[i].clip(max=0))
#        pos=pos*pos_num
#        neg=neg*neg_num
#        if i<=len(corpus_nv3):
#            count2=count2+(pos<=np.abs(neg))
#        else:
#            count2=count2+(pos>=np.abs(neg))
#
#accuracy2=count2/float(cl)
#print accuracy2
#
#count3=0
#count4=0
#count5=0
#count6=0
count2_1=0
count2_2=0
count2_3=0

for i in range(0,len(weight_f_1)):
    if len(weight_f_1[i])!=0:
        weight_f_1[i]=np.array(weight_f_1[i])
        l=[]
        for j in range(0,len(corp_para_1[i])):
            l.append(len(corp_para_1[i][j]))
        l=np.array(l)
        wei=weight_f_1[i]*l
        ma_x=np.max(wei)
        mi_n=np.min(wei)
        if i<=len(corpus_nv3_1):
            count2_1=count2_1+(ma_x<=np.abs(mi_n))
            out2_1[i]=(ma_x>=np.abs(mi_n))
            
        else:
            count2_1=count2_1+(ma_x>=np.abs(mi_n))
            out2_1[i]=(ma_x>=np.abs(mi_n))
            
acc2_1=count2_1/float(len(out2_1))
print 'accuracy2_1=',acc2_1

for i in range(0,len(weight_f_2)):
    if len(weight_f_2[i])!=0:
        weight_f_2[i]=np.array(weight_f_2[i])
        l=[]
        for j in range(0,len(corp_para_2[i])):
            l.append(len(corp_para_2[i][j]))
        l=np.array(l)
        wei=weight_f_2[i]*l
        ma_x=np.max(wei)
        mi_n=np.min(wei)
        if i<=len(corpus_nv3_2):
            count2_2=count2_2+(ma_x<=np.abs(mi_n))
            out2_2[i]=(ma_x>=np.abs(mi_n))
            
        else:
            count2_2=count2_2+(ma_x>=np.abs(mi_n))
            out2_2[i]=(ma_x>=np.abs(mi_n))

acc2_2=count2_2/float(len(out2_2))
print 'accuracy2_2=',acc2_2

for i in range(0,len(weight_f_3)):
    if len(weight_f_3[i])!=0:
        weight_f_3[i]=np.array(weight_f_3[i])
        l=[]
        for j in range(0,len(corp_para_3[i])):
            l.append(len(corp_para_3[i][j]))
        l=np.array(l)
        wei=weight_f_3[i]*l
        ma_x=np.max(wei)
        mi_n=np.min(wei)
        if i<=len(corpus_nv3_3):
            count2_3=count2_3+(ma_x<=np.abs(mi_n))
            out2_3[i]=(ma_x>=np.abs(mi_n))
            
        else:
            count2_3=count2_3+(ma_x>=np.abs(mi_n))
            out2_3[i]=(ma_x>=np.abs(mi_n))
            
acc2_3=count2_3/float(len(out2_3))
print 'accuracy2_3=',acc2_3

out1_f=out1_1+out1_2+out1_3
out2_f=out2_1+out2_2+out2_3

out1_ff=np.zeros(len(out1_f))
out2_ff=np.zeros(len(out2_f))

count1_1=0
count1_2=0
count1_3=0

count2_1=0
count2_2=0
count2_3=0
for i in range(0,len(out1_f)):
    if out1_f[i]>=2:
        out1_ff[i]=1
        
for i in range(0,len(out2_f)):
    if out2_f[i]>=2:
        out2_ff[i]=1

acc1=(len(yy_tr_1)-np.sum(np.abs(out1_ff-yy_tr_1)))/len(yy_tr_1)
acc2=(len(yy_tr_2)-np.sum(np.abs(out2_ff-yy_tr_2)))/len(yy_tr_2)

print 'the accuracy for max/min is', acc1

for i in range(0,len(out1_1)):
    count1_1+=out1_ff[i]==out1_1[i]
    count1_2+=out1_ff[i]==out1_2[i]
    count1_3+=out1_ff[i]==out1_3[i]
    
print 'the cont. of procedure in max/min is',count1_1/float(len(out1_1))
print 'the cont. of facts in max/min is',count1_2/float(len(out1_1))
print 'the cont. of law in max/min is',count1_3/float(len(out1_1))
    
print 'the accurcy for weighted sum is', acc2

for i in range(0,len(out2_1)):
    count2_1+=out2_ff[i]==out2_1[i]
    count2_2+=out2_ff[i]==out2_2[i]
    count2_3+=out2_ff[i]==out2_3[i]
    
print 'the cont. of procedure in weighted sum is',count2_1/float(len(out2_1))
print 'the cont. of facts in weighted sum is',count2_2/float(len(out2_1))
print 'the cont. of law in weighted sum is',count2_3/float(len(out2_1))
        
#accuracy3=count3/float(cl)
#accuracy4=count4/float(cl)
#print accuracy3 ,accuracy4
#print count5/float(len(corpus_nv3))
#print count6/float(len(corpus_v3))

'''
corpus_f=corpus_nv3+corpus_nv5+corpus_nv6+corpus_nv8+corpus_v3+corpus_v5+corpus_v6+corpus_v8
tf_vect=TfidfVectorizer(lowercase=True,stop_words='english',min_df=50,max_df=0.5,encoding='ascii')
tf=tf_vect.fit_transform(corpus_f).toarray()

#xx=int(tf.shape[0])
xx=8
#yy=8
#yy=np.zeros((xx,yy))

tr_l_nv3=int(0.8*len(corpus_nv3))
tr_l_nv5=int(0.8*len(corpus_nv5))
tr_l_nv6=int(0.8*len(corpus_nv6))
tr_l_nv8=int(0.8*len(corpus_nv8))
tr_l_v3=int(0.8*len(corpus_v3))
tr_l_v5=int(0.8*len(corpus_v5))
tr_l_v6=int(0.8*len(corpus_v6))
tr_l_v8=int(0.8*len(corpus_v8))

tt_l_nv3=len(corpus_nv3)-tr_l_nv3
tt_l_nv5=len(corpus_nv5)-tr_l_nv5
tt_l_nv6=len(corpus_nv6)-tr_l_nv6
tt_l_nv8=len(corpus_nv8)-tr_l_nv8
tt_l_v3=len(corpus_v3)-tr_l_v3
tt_l_v5=len(corpus_v5)-tr_l_v5
tt_l_v6=len(corpus_v6)-tr_l_v6
tt_l_v8=len(corpus_v8)-tr_l_v8

y_tr_nv3=np.zeros(tr_l_nv3)
y_tr_nv5=np.zeros(tr_l_nv5)
y_tr_nv6=np.zeros(tr_l_nv6)
y_tr_nv8=np.zeros(tr_l_nv8)
y_tr_v3=np.ones(tr_l_v3)
y_tr_v5=np.ones(tr_l_v5)
y_tr_v6=np.ones(tr_l_v6)
y_tr_v8=np.ones(tr_l_v8)

y_tr=np.concatenate((y_tr_nv3,y_tr_nv5,y_tr_nv6,y_tr_nv8,y_tr_v3,y_tr_v5,y_tr_v6,y_tr_v8,))
y_tt_nv3=np.zeros(tt_l_nv3)
y_tt_nv5=np.zeros(tt_l_nv5)
y_tt_nv6=np.zeros(tt_l_nv6)
y_tt_nv8=np.zeros(tt_l_nv8)
y_tt_v3=np.ones(tt_l_v3)
y_tt_v5=np.ones(tt_l_v5)
y_tt_v6=np.ones(tt_l_v6)
y_tt_v8=np.ones(tt_l_v8)
y_tt=np.concatenate((y_tt_nv3,y_tt_nv5,y_tt_nv6,y_tt_nv8,y_tt_v3,y_tt_v5,y_tt_v6,y_tt_v8,))

#yy=out_arr(0,len(corpus_nv3),0,yy)
#jj=len(corpus_nv3)+len(corpus_nv5)
#yy=out_arr(len(corpus_nv3),jj,1,yy)
#kk=jj+len(corpus_nv6)
#yy=out_arr(jj,kk,2,yy)
#jj=kk+len(corpus_nv8)
#yy=out_arr(kk,jj,3,yy)
#kk=jj+len(corpus_v3)
#yy=out_arr(jj,kk,4,yy)
#jj=kk+len(corpus_v5)
#yy=out_arr(kk,jj,5,yy)
#kk=jj+len(corpus_v6)
#yy=out_arr(jj,kk,6,yy)
#jj=kk+len(corpus_v8)
#yy=out_arr(kk,jj,7,yy)

#y1_tr=out_arr(xx,tr_l_nv3,0)
#y2_tr=out_arr(xx,tr_l_nv5,1)
#y3_tr=out_arr(xx,tr_l_nv6,2)
#y4_tr=out_arr(xx,tr_l_nv8,3)
#y5_tr=out_arr(xx,tr_l_v3,4)
#y6_tr=out_arr(xx,tr_l_v5,5)
#y7_tr=out_arr(xx,tr_l_v6,6)
#y8_tr=out_arr(xx,tr_l_v8,7)
#
#y1_tt=out_arr(xx,tt_l_nv3,0)
#y2_tt=out_arr(xx,tt_l_nv5,1)
#y3_tt=out_arr(xx,tt_l_nv6,2)
#y4_tt=out_arr(xx,tt_l_nv8,3)
#y5_tt=out_arr(xx,tt_l_v3,4)
#y6_tt=out_arr(xx,tt_l_v5,5)
#y7_tt=out_arr(xx,tt_l_v6,6)
#y8_tt=out_arr(xx,tt_l_v8,7)

y1_tr=out_arr(xx,len(corpus_nv3),0)
y2_tr=out_arr(xx,len(corpus_nv5),1)
y3_tr=out_arr(xx,len(corpus_nv6),2)
y4_tr=out_arr(xx,len(corpus_nv8),3)
y5_tr=out_arr(xx,len(corpus_v3),4)
y6_tr=out_arr(xx,len(corpus_v5),5)
y7_tr=out_arr(xx,len(corpus_v6),6)
y8_tr=out_arr(xx,len(corpus_v8),7)

yy_tr=np.concatenate((y1_tr,y2_tr,y3_tr,y4_tr,y5_tr,y6_tr,y7_tr,y8_tr))
#yy_tt=np.concatenate((y1_tt,y2_tt,y3_tt,y4_tt,y5_tt,y6_tt,y7_tt,y8_tt))

tr_l=tr_l_nv3+tr_l_nv5+tr_l_nv6+tr_l_nv8+tr_l_v3+tr_l_v5+tr_l_v6+tr_l_v8
x_tr=tf[0:tr_l]
x_tt=tf[tr_l:int(tf.shape[0])] 


clf=MLPRegressor(hidden_layer_sizes=500)
clf.fit(x_tr,yy_tr)
NN_pred=clf.predict(x_tt)

NN_out=np.zeros(NN_pred.shape)
for i in range(0, int(NN_pred.shape[0])):
    ind=np.argmax(NN_pred[i])
    NN_out[i][ind]=1
    
NN_accuracy=(int(tf.shape[0])-np.sum(np.abs(yy_tt-NN_out)))/100.0

print 'The accuracy of NN for BBoW is ' ,NN_accuracy

'''