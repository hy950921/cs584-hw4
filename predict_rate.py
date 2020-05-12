import numpy as np
import smart_open
from nltk.tokenize import word_tokenize
from utils.Reader import Reader
from utils.dataset import Dataset
from utils.SVDmodel import SVD
from time import time
import pandas as pd


K = 9
user_movie_rating = {}
movie_genres = {}
movie_directors = {}
movie_actors = {}
movie_tags = {}

def readFile(file, separator, columns, types):
  #print("cwd: %s" % os.getcwd())
  return pd.read_csv(file, sep=separator, names=columns, low_memory=False, dtype=types, header=0)

#General purpose - helps determine similarity between two movies by genres, director, actors, etc.
#Actually returns numerator and denominator for Dice's coefficient, so all similarities can
#be summed later.
def get_sim_of_feature(mID1, mID2, dct):
  feature1 = dct.get(mID1)
  feature2 = dct.get(mID2)
  if (feature1 == None and feature2 == None):
    return 0, 0
  if (feature1 == None):
    return 0, len(set(feature2))
  if (feature2 == None):
    return 0, len(set(feature1))
  m1Set = set(feature1)
  m2Set = set(feature2)
  return len(m1Set.intersection(m2Set)), len(m1Set) + len(m2Set)
  

def contentBased_predict(uid, mid):
  #list of IDs
  movie_ids = []
  #dict of IDs -> ratings
  movie_ratings = {}
  #list of <=K most similar movie IDs
  most_similar_ids = []
  for tup in user_movie_rating[uid]:
    movie_ids.append(tup[0])
    movie_ratings[tup[0]] = tup[1]
  
  for seenID in movie_ids:
    #add up intersections and total counts of features from movies mid and seenID
    genreSame, genreTotal = get_sim_of_feature(mid, seenID, movie_genres)
    directorSame, directorTotal = get_sim_of_feature(mid, seenID, movie_directors)
    actorSame, actorTotal = get_sim_of_feature(mid, seenID, movie_actors)
    tagSame, tagTotal = get_sim_of_feature(mid, seenID, movie_tags)

    #Calculate Dice's coefficient
    sameFeats = genreSame + directorSame + actorSame + tagSame
    totalFeats = genreTotal + directorTotal + actorTotal + tagTotal
    sim = 2 * (sameFeats / totalFeats)

    #see if this movie is more similar than current most similar movies
    if (len(most_similar_ids) < K):
      most_similar_ids.append( (seenID, sim) )
    elif (most_similar_ids[-1][1] < sim):
      most_similar_ids[-1] = (seenID, sim)
      most_similar_ids = sorted(most_similar_ids, key=lambda tup: tup[1])

  #At this point, we have a list of K most similar movies to movies that
  #the user has already seen.
  #Now we predict the rating of movie mid by averaging their ratings
  avg = 0
  for tup in most_similar_ids:
    avg += movie_ratings[tup[0]]
  avg /= len(most_similar_ids)

  #round to nearest tenth
  return round(avg, 1)


def fill_content_dict(file, dct):
  with smart_open.smart_open(file, "r", encoding='ISO-8859-1') as feature_rows:
    next(feature_rows)  # skip first line
    for row in feature_rows:
      tokens = word_tokenize(row)
      if (dct.get(int(tokens[0])) == None):
        dct[int(tokens[0])] = []
      dct[int(tokens[0])].append(tokens[1])
    feature_rows.close()


"""
  Predicts ratings for each user-movie in test file
"""


def cf_predict(train_data, test_data, cf_model):
  predictions = []
  print('Training cf model...')
  cf_model.fit(train_data)

  print('Predicting cf rates...')
  # Loop through each user-movie row in the test DataFrame
  for _, row in test_data.iterrows():
    userId = row['userID']
    movieId = row['movieID']
    prediction = cf_model.predict(userId, movieId)

    #roundedPrediction = str(round(prediction.est, 1))  # Round to nearest
    predictions.append(prediction)

  return predictions

def main():
  timeStart = time()
  #preprocess data

  print('Pre-processing cf data...')
  # Read train (shuffled) and test data as DataFrames
  train_data = readFile('./data/train.csv', separator=' ', columns=['userID', 'movieID', 'rating'],
                       types={'userID': np.int32, 'movieID': np.int32, 'rating': np.float32})
  train_data = train_data.sample(n=len(train_data))
  test_data = readFile('./data/test.csv', separator=' ', columns=['userID', 'movieID'],
                      types={'userID': np.int32, 'movieID': np.int32})

  # Build the train data as a Surprise's DataSet object
  reader = Reader(rating_scale=(0, 5))  # Standardized rating scale
  train_data = Dataset.load_from_df(train_data, reader)

  model = SVD(n_factors=5, n_epochs=50)

  # Build a Trainset object to feed into the prediction algorithm.
  train_data = train_data.build_full_trainset()

  # Predict ratings for each user and associated movie
  cf_predictions = cf_predict(train_data, test_data, model)

  print('\ncf train-predict costs (%d seconds)' % (time() - timeStart))

  #create dict {userID: [(movieID1, rating1), (movieID2, rating2), ...]}
  #only includes ratings for movies that the user has seen
  with open("./data/train.csv", "r") as data:
    data = data.readlines()
    for row in data[1:]:
      tokens = word_tokenize(row)
      if (user_movie_rating.get(int(tokens[0])) == None):
        user_movie_rating[int(tokens[0])] = []
      user_movie_rating[int(tokens[0])].append((int(tokens[1]), float(tokens[2])))

  #create dict {movieID: [genre1, genre2, ...]}
  fill_content_dict("./data/additional_files/movie_genres.data", movie_genres)

  #create dict {movieID: director}
  fill_content_dict("./data/additional_files/movie_directors.data", movie_directors)

  #create dict {movieID: [actor1, actor2, ...]}
  fill_content_dict("./data/additional_files/movie_actors.data", movie_actors)

  #create dict {movieID: [tag1, tag2, ...]}
  fill_content_dict("./data/additional_files/movie_tags.data", movie_tags)


  print("Done! (%d seconds)\n" % (time() - timeStart))

  #predict ratings
  print("Predicting...")
  timePred = time()
  content_predictions = []
  with open("./data/test.csv", "r") as testFile:
    for row in testFile.readlines()[1:]:
      tokens = word_tokenize(row)
      content_predictions.append(contentBased_predict(int(tokens[0]), int(tokens[1])))

  print("Done! (%d seconds)\n" % (time() - timePred))

  #write results to file
  print("Writing to file...")
  timeWrite = time()
  with open("./data/res.data", 'w') as predFile:
    for p1, p2 in zip(content_predictions, cf_predictions):
      predFile.write("%f\n" % ((float(p1)+float(p2))/2))

if __name__=="__main__":
  main()