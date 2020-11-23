# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 22:29:57 2018

@author: agabh
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 19:02:02 2018

@author: agabh
"""

import re
from io import open
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn import svm

#nv3='C:\\Users\\agabh\\Desktop\\NLP\\project\\data\\nv3'
nv5='C:\\Users\\agabh\\Desktop\\NLP\\project\\data\\nv5'
#nv6='C:\\Users\\agabh\\Desktop\\NLP\\project\\data\\nv6'
#nv8='C:\\Users\\agabh\\Desktop\\NLP\\project\\data\\nv8'
#v3='C:\\Users\\agabh\\Desktop\\NLP\\project\\data\\v3'
v5='C:\\Users\\agabh\\Desktop\\NLP\\project\\data\\v5'
#v6='C:\\Users\\agabh\\Desktop\\NLP\\project\\data\\v6'
#v8='C:\\Users\\agabh\\Desktop\\NLP\\project\\data\\v8'

def corpus(address):
    list_doc=os.listdir(address)
    corp=[]
    for i in range(0,len(list_doc)):
        
        strr=address+'\\'+list_doc[i]
#        print(strr)
        with open("%s"% strr,'r', encoding='utf8') as f:
            doc=f.read()
#        doc1 =extractFacts(doc)
#        if doc1!=-1:
        corp.append(doc)
    return corp
        
def out_arr(len1,len2,j):
    mat=np.zeros((len2,len1))
    for i in range(0,len2):
        mat[i][j]=1
    return mat


def extractFacts(data):
    pat=re.compile(r'\nPROCEDURE\n+')
    pat2=re.compile(r'\nTHE FACTS\n+')
    pat3=re.compile(r'\n+')
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

    p = re.compile(r'[0-9]+\.?')
    str1=data[ind+9:ind1-8]
    str1 = re.sub(p,"",str1)
    str1= re.sub(pat3,'\n',str1)
    if ind==-1:
        return -1
    else:
        return str1

#corpus_nv3=corpus(nv3)
corpus_nv5=corpus(nv5)
#corpus_nv6=corpus(nv6)
#corpus_nv8=corpus(nv8)
#corpus_v3=corpus(v3)
corpus_v5=corpus(v5)
#corpus_v6=corpus(v6)
#corpus_v8=corpus(v8)
corpus_f=corpus_nv5+corpus_v5
#corpus_f=corpus_nv3+corpus_nv5+corpus_nv6+corpus_nv8+corpus_v3+corpus_v5+corpus_v6+corpus_v8
tf_vect=TfidfVectorizer(lowercase=True,stop_words='english',min_df=50,max_df=0.5,encoding='ascii')
tf=tf_vect.fit_transform(corpus_f).toarray()
ff=tf_vect.get_feature_names()
#xx=int(tf.shape[0])
xx=8
#yy=8
#yy=np.zeros((xx,yy))


#y_tr_nv3=np.zeros(tr_l_nv3)
#y_tr_nv5=np.zeros(tr_l_nv5)
#y_tr_nv6=np.zeros(tr_l_nv6)
#y_tr_nv8=np.zeros(tr_l_nv8)
#y_tr_v3=np.ones(tr_l_v3)
#y_tr_v5=np.ones(tr_l_v5)
#y_tr_v6=np.ones(tr_l_v6)
#y_tr_v8=np.ones(tr_l_v8)
#
#y_tr=np.concatenate((y_tr_nv3,y_tr_nv5,y_tr_nv6,y_tr_nv8,y_tr_v3,y_tr_v5,y_tr_v6,y_tr_v8,))
#y_tt_nv3=np.zeros(tt_l_nv3)
#y_tt_nv5=np.zeros(tt_l_nv5)
#y_tt_nv6=np.zeros(tt_l_nv6)
#y_tt_nv8=np.zeros(tt_l_nv8)
#y_tt_v3=np.ones(tt_l_v3)
#y_tt_v5=np.ones(tt_l_v5)
#y_tt_v6=np.ones(tt_l_v6)
#y_tt_v8=np.ones(tt_l_v8)
#y_tt=np.concatenate((y_tt_nv3,y_tt_nv5,y_tt_nv6,y_tt_nv8,y_tt_v3,y_tt_v5,y_tt_v6,y_tt_v8,))

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

#y1_tr=out_arr(xx,len(corpus_nv3),0)
#y2_tr=out_arr(xx,len(corpus_nv5),1)
#y3_tr=out_arr(xx,len(corpus_nv6),2)
#y4_tr=out_arr(xx,len(corpus_nv8),3)
#y5_tr=out_arr(xx,len(corpus_v3),4)
#y6_tr=out_arr(xx,len(corpus_v5),5)
#y7_tr=out_arr(xx,len(corpus_v6),6)
#y8_tr=out_arr(xx,len(corpus_v8),7)
y1=np.zeros(len(corpus_nv5))
y2=np.ones(len(corpus_v5))
yy_tr=np.concatenate((y1,y2))
#yy_tr=np.concatenate((y1_tr,y2_tr,y3_tr,y4_tr,y5_tr,y6_tr,y7_tr,y8_tr))
print 'data done'
X_train, X_test, y_train, y_test = train_test_split(tf, yy_tr)


clf1=MLPRegressor(hidden_layer_sizes=(500,))
clf1.fit(X_train,y_train)
NN_pred=clf1.predict(X_test)

clf=svm.SVC(kernel='linear')
clf.fit(X_train,y_train)
svm_pred=clf.predict(X_test)

#svm_out=np.zeros(svm_pred.shape)
#for i in range(0, int(svm_pred.shape[0])):
#    ind=np.argmax(svm_pred[i])
#    svm_out[i][ind]=1
    
NN_accuracy=(int(tf.shape[0])-np.sum(np.abs(y_test-NN_pred)))/tf.shape[0]
svm_accuracy=(int(tf.shape[0])-np.sum(np.abs(y_test-svm_pred)))/tf.shape[0]
print 'The accuracy of NN  is ' ,NN_accuracy
print 'The accuracy of svm  is ' ,svm_accuracy
cff=clf.coef_
'''
np.savetxt('coeff.txt',cff)
vocab=[]
for i in range(0,len(ff)):
    vv=ff[i]
    vv=vv.encode('utf-8','ignore')
    vocab.append(vv)
    
f=open('vocab_f.txt','w')
f.write(vocab)
'''