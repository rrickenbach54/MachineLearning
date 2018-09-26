import java.util.Random;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.math.RoundingMode;
import java.text.DecimalFormat;

public class Perceptron extends SupervisedLearner {

    private DecimalFormat df = new DecimalFormat("#.###");// Used to format weight display with just 3 decimal places.
    private Random rand = new Random(); // Creates new random seed after every epoch for shuffling

    // Utility variables
    private double bias = 1;
    private double target;
    private double net;
    private double output;
    private double learningRate = 0.1;
    private double previousWeights[];
    private double weights[];
    private double sum = 0;
    private boolean weightsSet = false;
    private double lastRun = 0;
    private double currentRun;
    private int runCount = 0;
    private int timesWithoutChange = 0;
    private String missClass = ""; // Was only used for gathering miscalculation rate.

    public Perceptron(Random rand) {

    }

    @Override
    // Percepton learner created by Ryan Rickenbach
    public void train(Matrix features, Matrix labels) throws Exception {

        // Set weights to random variables between -1 & 1 only when they are not
        // initialized already.
        if (!weightsSet) {
            intializeWeights(features.cols());
        }

        currentRun = measureAccuracy(features, labels); // Get the accuracy of the current weights
        missClass += Double.toString(currentRun) + ",";// No longer used but left in this was used to get data for
        // misclassification rate
        runCount++;

        // For loop for learning
        for (int i = 0; i < features.rows(); i++) {
            target = labels.get(i, 0);
            calculateOutput(i, features);
            adjustWeights(i, features);
        }

        // Criteria for continued learning
        // Criteria:
        // Must run at least two times
        // If there is improvement of at more than 0.5 continue. If less than 5 times
        // then stop
        if (runCount < 2) {
            missClass += Double.toString(currentRun) + ",";
            lastRun = currentRun;
            train(features, labels);
        }
        // printWeights();
        else if (currentRun - lastRun > 0.5 && timesWithoutChange < 5) {
            missClass += Double.toString(currentRun) + ",";
            lastRun = currentRun;
            timesWithoutChange = 0;
            train(features, labels);
        } else if (currentRun - lastRun < 0.5 && timesWithoutChange < 5) {
            timesWithoutChange++;
            lastRun = currentRun;
            train(features, labels);
        } else {
            // Print run count and final weights.
            System.out.println("Final Run Count: " + runCount);
            System.out.println("Final Weights:");
            printWeights();
        }

        // Shuffle data after each epoch
        features.shuffle(rand, labels);

    }

    // Method to get the current accuracy with the given weights.
    private double measureAccuracy(Matrix features, Matrix labels) {

        int correct = 0;
        double percentCorrect = 0;
        for (int i = 0; i < features.rows(); i++) {
            target = labels.get(i, 0);
            calculateOutput(i, features);
            if (target == output) {
                correct++;
            }
        }
        percentCorrect = (double) correct / features.rows();
        return percentCorrect;
    }
//Methods that ultimately were never used. Allowed weights to step back if there was a degredation in accuracy
//	private void revertWeights() {
//
//		for(int i=0;i<weights.length;i++)
//		{
//			weights[i]=previousWeights[i];
//		}
//	}
//
//	private void storeWeights() {
//
//		for(int i=0;i<weights.length;i++)
//		{
//			previousWeights[i]=weights[i];
//		}
//	}

    // Method to adjust the weights of the perceptoron
    private void adjustWeights(int i, Matrix features) {


        if (target != output) {
            for (int j = 0; j < weights.length; j++) {
                if (j == weights.length - 1) {
                    double weight = weights[j];
                    double changeWeight = learningRate * (target - output) * bias;
                    weights[j] = (weight + changeWeight);
                } else {
                    double weight = weights[j];
                    double changeWeight = learningRate * (target - output) * features.get(i, j);
                    weights[j] = (weight + changeWeight);
                }

            }
        }
    }

    // Method for formatting and printing weights.
    private void printWeights() {

        df.setRoundingMode(RoundingMode.CEILING);
        System.out.println("Weights:");
        for (int i = 0; i < weights.length; i++) {
            System.out.print(df.format(weights[i]));
            if (i != weights.length - 1) {
                System.out.print(", ");
            }
        }
        System.out.println();
    }

    // Method for predicting the output.
    private void calculateOutput(int i, Matrix features) {

        sum = 0;

        for (int j = 0; j < features.cols() + 1; j++) {
            if (j == features.cols()) {
                sum += bias * weights[j];
            } else {
                sum += features.get(i, j) * weights[j];
            }
        }
        if (sum > 0) {
            output = 1;
        } else if (sum < 0) {
            output = 0;
        }
    }

    // Get the number of variables and create an n+1 weight array
    // Assigns a random double between -1 & 1
    private void intializeWeights(int cols) {
        weights = new double[cols + 1];
        previousWeights = new double[cols + 1];
        for (int i = 0; i < cols + 1; i++) {
            weights[i] = generateRand();
            previousWeights[i] = 0.0;
        }
        weightsSet = true;
    }

    // Utility method to create the random double.
    private double generateRand() {
        Random generator = new Random();
        int range = generator.nextInt(1000 + 1 - 100) - 100;
        double rand = (double) range / 1000;
        return rand;
    }

    @Override
    // Predict method called by SupervisedLearner.
    public void predict(double[] features, double[] labels) throws Exception {
        double net = 0;
        for (int i = 0; i < features.length + 1; i++) {
            if (i == features.length) {
                net += bias * weights[i];
            } else {
                net += features[i] * weights[i];
            }
        }
        if (net > 0) {
            labels[0] = 1;
        } else
            labels[0] = 0;
    }

}
