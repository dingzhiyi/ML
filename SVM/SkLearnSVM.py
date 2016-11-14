from sklearn import svm

X = [[2,0],[1,1],[2,3]]
y = [0,0,1]

clf = svm.SVC(kernel='linear')
clf.fit(X,y)

print(clf)
print('\n'+str(clf.support_vectors_))
print(str(clf.support_))
print(str(clf.n_support_))

print(clf.predict([[3,3]]))