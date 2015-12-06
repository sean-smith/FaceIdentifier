#/usr/bin/python
#takes two argumaents (other than the name of this file), the name of the results file, and the number of classes
#The results file should be tab delimited in the format: TRUE_CLASS_NUMBER, PREDICTED_CLASS_NUMBER
import sys

#helper functions
def generateZeroArray(length):
	list = []
	for i in range(length):
		list.append(0)
	return list



#arguments
results =open(sys.argv[1])
numClasses = int(sys.argv[2])
predictionPairs = {}
#determining results
for line in results:
	fields = line.split("\t")
	trueNumber = int(fields[0])
	predictedNumber = int(fields[1])
	if trueNumber in  predictionPairs:
		valList = predictionPairs.get(trueNumber);
		valList.append(predictedNumber)
		#append to the array
	else:
		predictionPairs[trueNumber] = [predictedNumber]
		#create key-val pair with an array

#Accumulate output in csv format
csv = ""
for i in range(numClasses):
	#get an empty array of size numClasses
	values = generateZeroArray(numClasses)
	#get the true number's values
	listOfPredictions = predictionPairs.get(i)
	#increase the values array at the corect index for each prediciton
	for j in range(len(listOfPredictions)):
		prediction = listOfPredictions[j]
		oldVal = values[prediction]
		newVal = oldVal + 1
		values[prediction] = newVal

	for val in values:
		csv += str(val)
		csv += ","
		#note - results in a hanging comma
	csv += "\n"

print csv
	#use values array to generate the csv format
	
