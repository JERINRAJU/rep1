from scipy.spatial import distance

#
#
def eucl(a, b):
	return distance.euclidean(a,b)
class myKNN():
	def fit(self, x_train,y_train):     #calssifier method
		self.x_train = x_train
		self.y_train = y_train

	def predict(self, x_train):   #predict method of classifier
		prediction = []
		for row in x_test:
			label = self.closest(row)
			prediction.append(label)
		return prediction

	def closest(self,row):
		best_dist = eucl(row,self.x_train[0])
		best_index = 0
		for i in range(1, len(self.x_train)):
			dist = eucl(row, self.x_train[i])
			if dist < best_dist:
				best_dist = dist
				best_index = 1
			return self.y_train[best_index]







from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

iris = load_iris()
features = iris.data
labels = iris.target

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size=.3)

clf = DecisionTreeClassifier()

clf.fit(x_train,y_train)
p = clf.predict(x_test)

from sklearn.metrics import accuracy_score
print("accuracy=",accuracy_score(y_test,p))


from sklearn.neighbors import KNeighborsClassifier
knn = myKNN()
knn.fit(x_train, y_train)
p = knn.predict(x_test)
 
print("accuracy=",accuracy_score(y_test,p))






from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train, y_train)
p = lr.predict(x_test)
 
print("accuracy=",accuracy_score(y_test,p))


