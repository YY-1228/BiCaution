from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('/users/Bicaution/PTM/stanford-corenlp-full-2018-02-27')
import nltk
import re
fp = open("./data/new.txt")
f = open('./data/new_output.txt','w')
stemmer = nltk.stem.SnowballStemmer('english') ;
wnl = nltk.stem.WordNetLemmatizer() ;
#stemmer = nltk.stem.PorterStemmer()

def func(sentence) :
    li = [{''},{''},{''}] #subj ,verb,obj,iobj
    res={} # map
    for ele in sentence:
        if(ele[0]=='ROOT'):
            li[0].add(ele[2]-1)
            res[ele[2]] = 1+1
        # if(re.search("subj",ele[0]) ) :
        #     li[0].add(ele[2]-1)
        #     res[ele[2]] = 0+1
        elif(re.search("dobj",ele[0]) ) : 
            li[1].add(ele[2]-1)
            res[ele[2]] = 2+1
        elif(re.search("iobj",ele[0]) ) : 
            li[2].add(ele[2]-1)
            res[ele[2]] = 3+1
        elif(re.search("obj",ele[0]) ) : 
            li[1].add(ele[2]-1)
            res[ele[2]] = 2+1
        else : pass
    for ele in sentence:
        if(ele[0]=='compound' and res.get(ele[1]) ) :
            li[res.get(ele[1])-2].add(ele[2]-1)
    return li

def check(ls):
    ch = {'a','the','that','which','what','is','are','do','can','you','it','when','where'}
    for it in ch:
        if(ls==it) : return 0
    return 1

def stem(str,ls):
    str = str.split(' ')
    ans = ["","",""]
    for i in range(3):
        for ele in ls[i]:
            if isinstance(ele,int) : 
                if ele >= len(str) : continue ;
                ans[i]=ans[i]+str[ele]+" " 
        words = ans[i].split(" ")
        stemmed =[]
        for word in words :
            # if i==0 :
            #     stemmed.append(wnl.lemmatize(word,'n') )
            if i==1 :
                stemmed.append(wnl.lemmatize(word,'v') )
            else :
                stemmed.append(stemmer.stem(word))
        ans[i]=" ".join(stemmed) ;   
    return ans ;    
class Pair : #set
    def __init__(self,reason = {0},result={0},id=0):
        self.reason=reason
        self.result=result
        self.id = id
    stem = []
    
line = " "
count=0
cnt=0
idx = dict()
init_idx = dict()
#num=1000000
num=100
while count<num:
    line = fp.readline()
    if not line:
        break
    count+=1
    line = line.strip()
    strs = line.split("\t")
    if(len(strs)==2):
        #data cleaning
        if(check(strs[0]) == 0 or check(strs[1]) == 0 ): continue;
        if not init_idx.get(strs[0]) :
             cnt+=1
             init_idx[strs[0]] = Pair({0},{0},cnt)
        if not init_idx.get(strs[1]) :
             cnt+=1
             init_idx[strs[1]] = Pair({0},{0},cnt)

        s = strs[0].replace(" . ","")
        str = nlp.dependency_parse(s)
        ls = func(str)
        ls = stem(s,ls)
        res1 = " ".join(ls)
        init_idx[strs[0]].stem = ls

        s = strs[1].replace(" . ","")
        str = nlp.dependency_parse(s)
        ls = func(str)
        ls = stem(s,ls)
        res2= " ".join(ls)
        init_idx[strs[1]].stem = ls
        if not idx.get(res1) :
            idx[res1] = Pair({0},{0},0)
        if not idx.get(res2) :
            idx[res2] = Pair({0},{0},0)

        init_idx[strs[0]].result.add(init_idx[strs[1]].id ) ;
        init_idx[strs[1]].reason.add(init_idx[strs[0]].id ) ;

        idx[res1].result.add(init_idx[strs[1]].id) ; 
        idx[res2].reason.add(init_idx[strs[0]].id) ;

count = 0; cnt=0 ; 
for it in init_idx.items() :
    count+=1
    id = it[1].id 
    print(id,file=f) 
    print(it[0],file=f) 
    #stem
    print(",".join(it[1].stem),file=f) ;
    #reason
    it[1].reason.discard(0) 
    #original reason
    for i in it[1].reason : print(i,end=' ',file=f)
    print("\n",end='',file=f) 
    #result
    it[1].result.discard(0) 
    #original result
    for i in it[1].result : print(i,end=' ',file=f)
    print("\n",end='',file=f) 

    # sentence = " ".join(it[1].stem)
    # idx[sentence].reason.discard(0)
    # idx[sentence].result.discard(0)
    #
    # for i in idx[sentence].reason : print(i,end=' ',file=f)
    # print("\n",end='',file=f)
    #
    # for i in idx[sentence].result : print(i,end=' ',file=f)
    # print("\n",end='',file=f)
    #
    # cnt+=len(idx[sentence].reason)
    # cnt+=len(idx[sentence].result)
    #
    # print("\n",end='',file=f)
    # print("\n",end='',file=f)

print(count)
print(cnt/2)
print("end")

fp.close()
f.close() 
nlp.close()
