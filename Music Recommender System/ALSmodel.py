""" PySpark ALS application
"""
from pyspark import SparkContext  # pylint: disable=import-error
from pyspark.mllib.recommendation import ALS, Rating  # pylint: disable=import-error

from pyspark.mllib.recommendation import ALS
import math

seed = 5L
iterations = 10
regularization_parameter = 0.1
ranks = [4, 8, 12]
errors = [0, 0, 0]
err = 0
tolerance = 0.02

min_error = float('inf')
best_rank = -1
best_iteration = -1
for rank in ranks:
    model = ALS.train(training_RDD, rank, seed=seed, iterations=iterations,
                      lambda_=regularization_parameter)
    predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    errors[err] = error
    err += 1
    print 'For rank %s the RMSE is %s' % (rank, error)
    if error < min_error:
        min_error = error
        best_rank = rank

print 'The best model was trained with rank %s' % best_rank


def main():
    """ Train and evaluate an ALS recommender.
    """
    # Set up environment
    sc = SparkContext("local[*]", "RecSys")

    # Load and parse the data
    data = sc.textFile("./data/ratings.dat")
    ratings = data.map(parse_rating)

    # Build the recommendation model using Alternating Least Squares
    rank = 10
    iterations = 20
    model = ALS.train(ratings, rank, iterations)

    movies = sc.textFile("./data/songs.dat")\
               .map(parse_movie)
    # Evaluate the model on training data
    testdata = ratings.map(lambda p: (p[0], p[1]))
    predictions = model.predictAll(testdata)\
                       .map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = ratings.map(lambda r: ((r[0], r[1]), r[2]))\
                             .join(predictions)
    MSE = rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
    print("Mean Squared Error = " + str(MSE))


if __name__ == "__main__":
    main()