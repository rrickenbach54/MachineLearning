import java.util.Arrays;

public class InstanceBasedLearner extends SupervisedLearner
{
    private double[] kSmallest;//Holds the values of K smallest values only needed for weighted and regression
    private Matrix featuresMatrix;
    private Matrix labelsMatrix;
    private int k = 3; //Number of nearest neighbors
    private int[] knn; //Index array for where the k nearest neighbors exist, used for label lookups
    private double[] output; //Output array for predict
    private boolean regression = false; //Flag for code to use regression or classification
    private double regressionOutput; //Single value for regression output
    private boolean weighted = true; //Use weighted or unweighted this is a manual value change

    @Override
    public void train(Matrix features, Matrix labels)
    {
        //Allocate Array Memory
        populateArray();
        if (labels.valueCount(0) == 0)
        {
            regression = true;
        }
        if (!regression)
        {
            populateOutput(labels.valueCount(0));
        }
        featuresMatrix = features;
        labelsMatrix = labels;

    }

    private void populateOutput(int valueCount)
    {
        //Builds the output array to be the number of classification outputs never created with regression
        output = new double[valueCount];
    }

    private void populateArray()
    {
        //Build all the global arrays with correct dimensionality
        kSmallest = new double[k];
        knn = new int[k];
    }

    @Override
    public void predict(double[] features, double[] labels)
    {
        //Fill smallest distance with max value
        for (int i = 0; i < kSmallest.length; i++)
        {
            kSmallest[i] = Double.MAX_VALUE;
        }

        //Loops through and calculates the distances to each point. After it has the distance it checks to see if it is smaller than the largest nearest neighbor and replaces it if so
        //This originally stored all distances but after code cleanup it was not needed.
        for (int i = 0; i < featuresMatrix.rows(); i++)
        {
            double distance = 0;
            for (int j = 0; j < features.length; j++)
            {
                //Manhatten distance function
                distance += Math.abs(featuresMatrix.get(i, j) - features[j]);
            }
            double largestSmall = -1;
            int index = 0;
            //Flag where the largest nearest neighbor is located and if distance is smaller replace it
            for (int k = 0; k < kSmallest.length; k++)
            {
                if (kSmallest[k] > largestSmall)
                {
                    largestSmall = kSmallest[k];
                    index = k;
                }

            }
            if (distance < largestSmall)
            {
                kSmallest[index] = distance;
                knn[index] = i; //Sets a pointer to the feature for label lookup later
            }
        }

        //Get class of knn of classification
        if (!regression)
        {
            //Weighted
            if (weighted)
            {
                for (int i = 0; i < knn.length; i++)
                {
                    //Common function for distance squared and error checking for div by 0 errors
                    double distanceSquared = Math.pow(kSmallest[i], 2);
                    if (distanceSquared != 0)
                    {
                        output[(int) labelsMatrix.get(knn[i], 0)] += (1 / distanceSquared);
                    }
                }
                double denominator = 0;
                for (double aKSmallest : kSmallest)
                {
                    double distanceSquared = Math.pow(aKSmallest, 2);
                    if (distanceSquared != 0)
                    {
                        denominator += (1 / distanceSquared);
                    }
                }

                if (denominator != 0)
                {
                    for (int i = 0; i < output.length; i++)
                    {
                        output[i] = (output[i] / denominator);
                    }
                }

                double greatestValue = 0;
                int predict = 0;
                //Predict based on output
                for (int i = 0; i < output.length; i++)
                {
                    if (output[i] > greatestValue)
                    {
                        greatestValue = output[i];
                        predict = i;
                    }
                }
                labels[0] = predict;
            }

            //Unweighted
            else
            {
                for (int aKnn : knn)
                {
                    output[(int) labelsMatrix.get(aKnn, 0)]++;
                }

                double greatestValue = 0;
                int predict = -1;
                //Predict based on output
                for (int i = 0; i < output.length; i++)
                {
                    if (output[i] > greatestValue)
                    {
                        greatestValue = output[i];
                        predict = i;
                    }
                }

                //WipeMemory
                Arrays.fill(output, 0);

                //Predict
                labels[0] = predict;

            }

        }


        //Calculate regression output
        if (regression)
        {
            double average = 0;
            double denominator = 0;
            if (weighted)
            {

                for (double aKSmallest : kSmallest)
                {
                    double distanceSquared = Math.pow(aKSmallest, 2);
                    if (distanceSquared != 0)
                    {
                        denominator += 1 / distanceSquared;
                    }
                }
                double numerator = 0;
                for (int i = 0; i < knn.length; i++)
                {
                    double distanceSquared = Math.pow(kSmallest[i], 2);
                    if (distanceSquared != 0)
                    {
                        numerator += (1 / distanceSquared) * labelsMatrix.get(knn[i], 0);
                    }
                }
                if(denominator!=0)
                {
                    regressionOutput = numerator / denominator;
                }
                else
                {
                    regressionOutput=0;
                }
            } else
            {
                for (int aKnn : knn)
                {
                    average += labelsMatrix.get(aKnn, 0);
                }
                regressionOutput = average / k;
            }
            labels[0] = regressionOutput;
        }
    }
}
