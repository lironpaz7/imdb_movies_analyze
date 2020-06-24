import sys
from algorithm_runner import AlgorithmRunner
from algorithm_runner_race import AlgorithmRunnerRace
from data import Data
from data_race import DataRace

# Boolean variable to indicate which question to run
Q1 = True
Q2 = True


def main(argv):
    if Q1:
        data = Data(argv[1])
        data.preprocess()
        print("Question 1:")

        # ----------------- KNN -----------------
        knn = AlgorithmRunner("KNN")
        knn.run(data, knn)

        # ----------------- Rocchio -----------------
        rocchio = AlgorithmRunner("Rocchio")
        rocchio.run(data, rocchio)
        print()

    if Q2:
        data2 = DataRace(argv[1])
        data2.preprocess()
        print("Question 2:")

        # ----------- testing best k for KNN from (2,3,...19,20) ------------
        # best_k = 0
        # max_accuracy = 0
        # for k in range(2, 21):
        #     knn = AlgorithmRunnerRace("KNN", k)
        #     tmp = knn.run(data2, knn)
        #     if tmp > max_accuracy:
        #         max_accuracy = tmp
        #         best_k = k
        # print(f"highest accuracy was: {max_accuracy} with k={best_k}")

        # ----------------- KNN -----------------
        knn = AlgorithmRunnerRace("KNN", 11)
        knn.run(data2, knn)

        # ----------------- Rocchio -----------------
        rocchio = AlgorithmRunnerRace("Rocchio")
        rocchio.run(data2, rocchio)


if __name__ == '__main__':
    main(sys.argv)
