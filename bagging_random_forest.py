#-------------------------------------------------------------------------
# AUTHOR: Anita Mehrazarin
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #4
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn import tree
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
import csv

dbTraining = []
dbTest = []
X_training = []
y_training = []
CLASSIFER_VOTES = [] #this array will be used to count the votes of each classifier

#reading the training data from a csv file and populate dbTraining
#--> add your Python code here
with open('optdigits.tra', 'r') as trainingFile:
  reader = csv.reader(trainingFile)
  for i, row in enumerate(reader):
      dbTraining.append (row)

#reading the test data from a csv file and populate dbTest
#--> add your Python code here
with open('optdigits.tes', 'r') as testingFile:
  reader = csv.reader(testingFile)
  for i, row in enumerate(reader):
      dbTest.append (row)
      CLASSIFER_VOTES.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

#inititalizing the class votes for each test sample. Example: classVotes.append([0,0,0,0,0,0,0,0,0,0])
#--> add your Python code here

print("Started my base and ensemble classifier ...")

classPredict = []
classPredict_rf = []

for k in range(20): #we will create 20 bootstrap samples here (k = 20). One classifier will be created for each bootstrap sample

  bootstrapSample = resample(dbTraining, n_samples=len(dbTraining), replace=True)

  #populate the values of X_training and y_training by using the bootstrapSample
  #--> add your Python code here
  for i in range(len(bootstrapSample)):
      X_training.append(bootstrapSample[i][:65])

  for i in range(len(bootstrapSample)):
      y_training.append(bootstrapSample[i][64])

  #fitting the decision tree to the data
  clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=None) #we will use a single decision tree without pruning it
  clf = clf.fit(X_training, y_training)

  for i, testSample in enumerate(dbTest):

      #make the classifier prediction for each test sample and update the corresponding index value in classVotes. For instance,
      # if your first base classifier predicted 2 for the first test sample, then classVotes[0,0,0,0,0,0,0,0,0,0] will change to classVotes[0,0,1,0,0,0,0,0,0,0].
      # Later, if your second base classifier predicted 3 for the first test sample, then classVotes[0,0,1,0,0,0,0,0,0,0] will change to classVotes[0,0,1,1,0,0,0,0,0,0]
      # Later, if your third base classifier predicted 3 for the first test sample, then classVotes[0,0,1,1,0,0,0,0,0,0] will change to classVotes[0,0,1,2,0,0,0,0,0,0]
      # this array will consolidate the votes of all classifier for all test samples
      #--> add your Python code here
      class_predicted = int(clf.predict([dbTest[i][:65]])[0])
      classPredict.append(class_predicted)
      CLASSIFER_VOTES[i][class_predicted] += 1

  if k == 0: #for only the first base classifier, compare the prediction with the true label of the test sample here to start calculating its accuracy
         #--> add your Python code here
         CORRECT = 0
         ACCURACY = 0
         for j in range(len(dbTest)):
             if class_predicted == int(dbTest[j][64]):
                 CORRECT += 1

  ACCURACY = CORRECT / len(dbTest)

  if k == 0: #for only the first base classifier, print its accuracy here
     #--> add your Python code here
     print("Finished my base classifier (fast but relatively low accuracy) ...")
     print("My base classifier accuracy: " + str(ACCURACY))
     print("")

  #now, compare the final ensemble prediction (majority vote in classVotes) for each test sample with the ground truth label to calculate the accuracy of the ensemble classifier (all base classifiers together)
  #--> add your Python code here
  maxVal = 0
  val = 0
  vals = []
  CORRECT = 0
  ACCURACY = 0

  for j in range(len(dbTest)):
      for k in range(10):
          if k == 0:
              maxVal = CLASSIFER_VOTES[j][k]
          if CLASSIFER_VOTES[j][k] > maxVal:
              maxVal = CLASSIFER_VOTES[j][k]
              val = CLASSIFER_VOTES[j].index(CLASSIFER_VOTES[j][k])
          vals.append(val)

  for j in range(len(dbTest)):
      if vals[j] == int(dbTest[j][64]):
          CORRECT += 1

  ACCURACY = CORRECT / len(dbTest)

#printing the ensemble accuracy here
print("Finished my ensemble classifier (slow but higher accuracy) ...")
print("My ensemble accuracy: " + str(ACCURACY))
print("")

print("Started Random Forest algorithm ...")

#Create a Random Forest Classifier
clf=RandomForestClassifier(n_estimators=20) #this is the number of decision trees that will be generated by Random Forest. The sample of the ensemble method used before

#Fit Random Forest to the training data
clf.fit(X_training,y_training)

#make the Random Forest prediction for each test sample. Example: class_predicted_rf = clf.predict([[3, 1, 2, 1, ...]]
#--> add your Python code here
for i in range(len(dbTest)):
      class_predicted_rf = int(clf.predict([dbTest[i][:65]])[0])
      classPredict_rf.append(class_predicted_rf)

#compare the Random Forest prediction for each test sample with the ground truth label to calculate its accuracy
#--> add your Python code here
CORRECT = 0
ACCURACY = 0
for j in range(len(dbTest)):
    if classPredict_rf[j] == int(dbTest[j][64]):
        CORRECT += 1
ACCURACY = CORRECT / len(dbTest)

#printing Random Forest accuracy here
print("Random Forest accuracy: " + str(ACCURACY))

print("Finished Random Forest algorithm (much faster and higher accuracy!) ...")
