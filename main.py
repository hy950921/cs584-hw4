# Library imports
import numpy as np
import smart_open
from sklearn.pipeline import Pipeline
from surprise import SVD, Dataset, Reader
from time import time
# Custom file imports
from crossvalidation.crossvalidation import computeCVAverageRMSE
from preprocess.fileutil import readFile

def main():
  startTime = time()
  print('Pre-processing...')
  # Read train (shuffled) and test data as DataFrames
  trainData = readFile('./data/train.data', separator=' ', columns=['userID', 'movieID', 'rating'], types={'userID': np.int32, 'movieID': np.int32, 'rating': np.float32})
  trainData = trainData.sample(n=len(trainData))
  testData = readFile('./data/test.data', separator=' ', columns=['userID', 'movieID'], types={'userID': np.int32, 'movieID': np.int32})

  # Build the train data as a Surprise's DataSet object
  reader = Reader(rating_scale=(0, 5)) # Standardized rating scale 
  trainData = Dataset.load_from_df(trainData, reader)
  
  predictionAlgorithm = SVD(n_factors=5, n_epochs=50)

  """
  # Cross validation for k=5
  avgRMSE = computeCVAverageRMSE(trainData, predictionAlgorithm)
  print(avgRMSE)
  return
  """

  # Build a Trainset object to feed into the prediction algorithm.
  trainData = trainData.build_full_trainset()

  # Predict ratings for each user and associated movie
  predictions = predictRatings(trainData, testData, predictionAlgorithm)

  # Write the predictions to a file
  writePredictions(predictions)
  
  print('\nUser-movie ratings successfully written to predictions.data (%d seconds)' % (time() - startTime))

"""
  Predicts ratings for each user-movie in test file
"""
def predictRatings(trainData, testData, predictionAlgorithm):
  predictions = []
  print('Training...')
  predictionAlgorithm.fit(trainData)

  print('Predicting...')
  # Loop through each user-movie row in the test DataFrame
  for _, row in testData.iterrows():
    userId = row['userID']
    movieId = row['movieID']
    prediction = predictionAlgorithm.predict(userId, movieId)

    roundedPrediction = str(round(prediction.est, 1)) # Round to nearest
    predictions.append(roundedPrediction)

  return predictions


"""
  Writes predictions to a file.
"""
def writePredictions(predictions):
  print('Writing predictions...')
  with smart_open.smart_open("./data/predictions.data", "w") as f:
    for prediction in predictions:
        s = prediction+"\n"
        f.write(s)
    f.close()

if __name__ == '__main__':
  main()
