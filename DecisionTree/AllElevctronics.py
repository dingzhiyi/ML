from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import tree
from sklearn import preprocessing

allElectronicsData = open('D:/Code/DeepLearningBasics/DecisionTree/AllElectronics.csv','r')
reader = csv.reader(allElectronicsData)
headers = next(reader)

labelList = []
featureList = []

for row in reader:
    labelList.append(row[len(row)-1])
    rowDict = {}
    for i in range(1,len(row)-1):
        rowDict[headers[i]]=row[i]
    featureList.append(rowDict)


    
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()
print(dummyX)
print(vec.get_feature_names())

lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print('dummyY: '+str(dummyY))

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX,dummyY)
print("clf: "+str(clf))

with open("D:/Code/DeepLearningBasics/DecisionTree/allElectronicInformationGainOri.dot",'w') as f:
    f = tree.export_graphviz(clf,feature_names=vec.get_feature_names(),out_file=f)
    
oneRowX = dummyX[0,:]
print(oneRowX)
newRowX = oneRowX
newRowX[0] = 1
newRowX[2] = 0

predictedY = clf.predict(newRowX.reshape(1,-1))
print(str(predictedY))    

