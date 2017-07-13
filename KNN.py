from scipy.io import mmread #mmread is used to read matrix file
import re
import random
import math
import operator
 
#Cosine function where cosine similiarity is calculated
def cosine(settest,settrain,vec1,vec2): 

    intersection=list(settrain & settest)
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    if not denominator:
        return 0.0
    else:
        return round(float(numerator) / denominator, 3)
    
#Accuracy fucntion where the accuracy is calculated in perentage        
def getAccuracy(testSet, predictions): 
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0


#Weighting Scheme is implemented here by inversing the cosine value 
#and adding them for each separate class

def getWeightedResponse(neighbors):
    weights = {}
    for x in neighbors:
        for elem in x:
            dist=elem[1]
            if dist!=0:
                weight=1/dist
            else:
                weight=0
                
            element=elem[0]
            classvariable=element[3]
            if classvariable in weights:
                weights[classvariable] += weight
            else:
                weights[classvariable] = weight
    sortedWeights=sorted(weights.items(), key=operator.itemgetter(1), reverse=True)
    return sortedWeights[0][0]
        
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        data = neighbors[x][0]
        temp=data[0]
        response=temp[3]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]
#The K number of neighbors is got here by descending order of the weights     
def getNeighbors(training_dict,test_dict,trainingSet, testInstance, k,vec1,vec2):
    temp = []
    distances=[]
    settest=set(test_dict[testInstance[0]])
    for x in trainingSet:
        settrain=set(training_dict[x[0]])
        dist=cosine(settest, settrain,vec1,vec2)
        temp.append((x, dist))
    for c in temp:
        distances.append(list(c))
    distances.sort(key=operator.itemgetter(1),reverse=True)
    neighbors = []
    neighbors.append(distances[:k])
    return neighbors

#This method helps us find the accuracy of the Unweighted accuracy    
def getUnweigtedAccuracy(trainingSet,testSet,k,vec1,vec2):
    training_dict={}
    test_dict={}
    for elem in trainingSet:
        if elem[0] in training_dict.keys():
                training_dict[elem[0]].append(elem[1])
        else:
                training_dict[elem[0]] = [elem[1]]
                
    for elem in testSet:
        if elem[0] in test_dict.keys():
                test_dict[elem[0]].append(elem[1])
        else:
                test_dict[elem[0]] = [elem[1]]
    #process_docs(trainingSet)
    predictions=[]
    for x in range(len(testSet)):
        neighbors = getNeighbors(training_dict,test_dict,trainingSet, testSet[x], k,vec1,vec2)
        result = getResponse(neighbors)
        predictions.append(result)
        print('predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print("Accuracy of UnWeighted Classifier")
    print('Accuracy: ' + repr(accuracy) + '%')
    
#This method helps us find the accuracy of the nweighted accuracy    
def getWeigtedAccuracy(trainingSet,testSet,k,vec1,vec2):
    training_dict={}
    test_dict={}
    for elem in trainingSet:
        if elem[0] in training_dict.keys():
                training_dict[elem[0]].append(elem[1])
        else:
                training_dict[elem[0]] = [elem[1]]
                
    for elem in testSet:
        if elem[0] in test_dict.keys():
                test_dict[elem[0]].append(elem[1])
        else:
                test_dict[elem[0]] = [elem[1]]
    predictions=[]
    for x in range(len(testSet)):
        neighbors = getNeighbors(training_dict,test_dict,trainingSet, testSet[x], k,vec1,vec2)
        result = getWeightedResponse(neighbors)
        predictions.append(result)
        print('predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print("Accuracy of weighted Classifier")
    print('Accuracy: ' + repr(accuracy) + '%')
   
    
def main():
    mat=mmread('news_articles.mtx')#Reading the matrix file into the program
    processedDataset=[]
    rawdata=[]
    trainingSet=[]
    testSet=[]
    k=3  #K value is set here

    text=open('news_articles.txt') #Labels file is read here

    for x in text:
        f=re.findall(r"[\w']+|[,]", x)
        rawdata.append(f)
    i=0
    while(i<221029): #the "i" value is the number of rows and columns and data in the matrix
        for x in rawdata:
            temp=[]
            if(str(mat.row[i])==str(x[0])):
                temp.append(str(mat.row[i]))
                temp.append(str(mat.col[i]))
                temp.append(str(mat.data[i]))
                temp.append(x[2])
                processedDataset.append(temp)
        i=i+1
    for x in range(len(processedDataset)-1):
            for y in range(3):
                processedDataset[x][y] = str(processedDataset[x][y])
            if random.random()<0.66:
                trainingSet.append(processedDataset[x])
            else:
                testSet.append(processedDataset[x])
    print('Train: ' + repr(len(trainingSet)))
    print( 'Test: ' + repr(len(testSet)))
    
    vec1 ={}
    for x in trainingSet:
        if x[1] in vec1.keys():
            vec1[x[1]]+=float(x[2])
        else:
            vec1[x[1]]=float(x[2])
    vec2 ={}
    for x in testSet:
        if x[1] in vec2.keys():
            vec2[x[1]]+=float(x[2])
        else:
            vec2[x[1]]=float(x[2])
    #getUnweigtedAccuracy(trainingSet,testSet,k,vec1,vec2)#Method for getting accuracy of unweighted scheme
    getWeigtedAccuracy(trainingSet,testSet,k,vec1,vec2)#Method for getting accuracy of weighted scheme
    
main()