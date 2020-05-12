import numpy as np
import smart_open
from time import time
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess.fileutil import readFile

K = 9
userRatings = {}
movieGenres = {}
movieDirectors = {}
movieActors = {}
movieTags = {}

#General purpose - helps determine similarity between two movies by genres, director, actors, etc.
#Actually returns numerator and denominator for Dice's coefficient, so all similarities can
#be summed later.
def similarity(mID1, mID2, dct):
  gens1 = dct.get(mID1)
  gens2 = dct.get(mID2)
  if (gens1 == None and gens2 == None):
    return 0, 0
  if (gens1 == None):
    return 0, len(set(gens2))
  if (gens2 == None):
    return 0, len(set(gens1))
  m1Set = set(gens1)
  m2Set = set(gens2)
  return len(m1Set.intersection(m2Set)), len(m1Set) + len(m2Set)
  

def predictRating(uid, mid):
  #list of IDs
  seenMovieIDs = []
  #dict of IDs -> ratings
  seenMovieRatings = {}
  #list of <=K most similar movie IDs
  mostSimilarIDs = []
  for tup in userRatings[uid]:
    seenMovieIDs.append(tup[0])
    seenMovieRatings[tup[0]] = tup[1]
  
  for seenID in seenMovieIDs:
    #add up intersections and total counts of features from movies mid and seenID
    genreSame, genreTotal = similarity(mid, seenID, movieGenres)
    directorSame, directorTotal = similarity(mid, seenID, movieDirectors)
    actorSame, actorTotal = similarity(mid, seenID, movieActors)
    tagSame, tagTotal = similarity(mid, seenID, movieTags)

    #Calculate Dice's coefficient
    sameFeats = genreSame + directorSame + actorSame + tagSame
    totalFeats = genreTotal + directorTotal + actorTotal + tagTotal
    sim = 2 * (sameFeats / totalFeats)

    #see if this movie is more similar than current most similar movies
    if (len(mostSimilarIDs) < K):
      mostSimilarIDs.append( (seenID, sim) )
    elif (mostSimilarIDs[-1][1] < sim):
      mostSimilarIDs[-1] = (seenID, sim)
      mostSimilarIDs = sorted(mostSimilarIDs, key=lambda tup: tup[1])

  #At this point, we have a list of K most similar movies to movies that
  #the user has already seen.
  #Now we predict the rating of movie mid by averaging their ratings
  avg = 0
  for tup in mostSimilarIDs:
    avg += seenMovieRatings[tup[0]]
  avg /= len(mostSimilarIDs)

  #round to nearest tenth
  return round(avg, 1)

def fillDict(file, dct):
  with smart_open.smart_open(file, "r", encoding='ISO-8859-1') as fRows:
    next(fRows) #skip first line
    for row in fRows:
      tokens = word_tokenize(row)
      if (dct.get(int(tokens[0])) == None):
        dct[int(tokens[0])] = []
      dct[int(tokens[0])].append(tokens[1])
    fRows.close()

def main():
  timeStart = time()
  #preprocess data
  print("Preprocessing...")

  #create dict {userID: [(movieID1, rating1), (movieID2, rating2), ...]}
  #only includes ratings for movies that the user has seen
  with smart_open.smart_open("./data/train.data", "r") as gRows:
    next(gRows) #skip first line
    for row in gRows:
      tokens = word_tokenize(row)
      if (userRatings.get(int(tokens[0])) == None):
        userRatings[int(tokens[0])] = []
      userRatings[int(tokens[0])].append( (int(tokens[1]), float(tokens[2])) )
    gRows.close()

  #create dict {movieID: [genre1, genre2, ...]}
  fillDict("./data/additional_files/movie_genres.data", movieGenres)
  
  
  #create dict {movieID: director}
  fillDict("./data/additional_files/movie_directors.data", movieDirectors)

  #create dict {movieID: [actor1, actor2, ...]}
  fillDict("./data/additional_files/movie_genres.data", movieActors)
  
  #create dict {movieID: [tag1, tag2, ...]}
  fillDict("./data/additional_files/movie_genres.data", movieTags)


  print("Done! (%d seconds)\n" % (time() - timeStart))

  #predict ratings
  print("Predicting...")
  timePred = time()
  predictions = []
  with smart_open.smart_open("./data/test.data", "r") as testFile:
    next(testFile) #skip first line
    for row in testFile:
      tokens = word_tokenize(row)
      predictions.append(predictRating(int(tokens[0]), int(tokens[1])))
  
  print("Done! (%d seconds)\n" % (time() - timePred))

  #write results to file
  print("Writing to file...")
  timeWrite = time()
  with smart_open.smart_open("./data/predictions2.data", 'w') as predFile:
    for pred in predictions:
      predFile.write("%s\n" % str(pred))
  
  print("Done! (%d seconds)\n" % (time() - timeWrite))
  print("Completed in %d seconds" % (time() - timeStart))

if __name__=="__main__":
  main()