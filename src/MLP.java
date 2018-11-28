import java.util.Random;


class nLayer
{
    int size;
    double[] output;
    double[] error;
    double[][] weightChange;
    double[][] weightStored;
    double[][] weight;
}

public class MLP extends SupervisedLearner
{

    private double maxEpochs = 10000;
    private double learningRate = 0.05;
    private double momentum = 0.3;
    private int epochs = 1;
    private double bias = 1;
    private boolean validate = true;
    private boolean continueTraining = true;
    private double validationAccuracy = 0;
    private double previousValidationAccuracy = 0;
    private double maxAccuracy = 0;
    private int runsWithoutImprovement = 0;
    private int inputLayerSize;
    private int hiddenLayerSize;
    private int outputLayerSize;
    private nLayer Layer[];
    private Matrix globalFeatures;
    private Matrix globalLabels;

    private int numberOfLayers = 4;

    MLP()
    {
        super();
    }


    public void train(Matrix inputFeatures, Matrix inputLabels)
    {
        //Make the Matrix global for usage in other methods as needed without passing them.
        globalFeatures = inputFeatures;
        globalLabels = inputLabels;

        //Add 1 for bias node
        inputLayerSize = globalFeatures.cols() + 1;
        double hiddenLayerScale = 2;
        hiddenLayerSize = (int) hiddenLayerScale * inputLayerSize;
        outputLayerSize = globalLabels.valueCount(0);

        //Initialized Memory and build the hidden layers
        createLayers();
        intializeWeights();

        while ((epochs < maxEpochs) && continueTraining)
        {
            if (validate)
            {
                for (int i = 0; i < (int) (.75 * globalFeatures.rows() + 1); i++)
                {
                    populateInputLayer(i);
                    feedForward();
                    backprop(globalLabels.get(i, 0));
                }
            } else
            {
                for (int i = 0; i < globalFeatures.rows(); i++)
                {
                    populateInputLayer(i);
                    feedForward();
                    backprop(globalLabels.get(i, 0));
                }
            }

            if (validate)
            {
                calculateValidationAccuracy();
                if ((validationAccuracy - previousValidationAccuracy) > 0.01)
                {
                    previousValidationAccuracy = validationAccuracy;
                    runsWithoutImprovement = 0;
                    if (validationAccuracy > maxAccuracy)
                    {
                        //Store the best weights
                        storeWeights();
                        maxAccuracy = validationAccuracy;
                        if (validationAccuracy == 100)
                        {
                            continueTraining = false;
                        }
                    }
                } else
                {
                    runsWithoutImprovement++;
                    previousValidationAccuracy = validationAccuracy;
                    if (runsWithoutImprovement >= 7)
                    {
                        continueTraining = false;
                    }
                }
            }
            globalFeatures.shuffle(new Random(), globalLabels);
            epochs++;
        }
        if (validationAccuracy < maxAccuracy)
        {
            //Restore weights to best validation set accuracy
            restoreWeights();
            validationAccuracy = maxAccuracy;
        }
        System.out.println("Training Completed");
        System.out.println("Total Epochs: " + epochs);
        System.out.println("Final Validation Accuracy: " + validationAccuracy);


    }

    private void restoreWeights()
    {
        //Restore best weights
        for (int i = 0; i < Layer.length - 1; i++)
        {
            for (int j = 0; j < Layer[i].weight.length; j++)
            {
                if (Layer[i].weight[0].length >= 0)
                    System.arraycopy(Layer[i].weightStored[j], 0, Layer[i].weight[j], 0, Layer[i].weight[0].length);
            }
        }
    }

    private void storeWeights()
    {
        //Store Weights
        for (int i = 0; i < Layer.length - 1; i++)
        {
            for (int j = 0; j < Layer[i].weight.length; j++)
            {
                if (Layer[i].weight[0].length >= 0)
                    System.arraycopy(Layer[i].weight[j], 0, Layer[i].weightStored[j], 0, Layer[i].weight[0].length);
            }
        }
    }

    private void calculateValidationAccuracy()
    {
        int numCorrect = 0;
        for (int i = (int) (.75 * globalFeatures.rows()); i < globalFeatures.rows(); i++)
        {
            populateInputLayer(i);
            feedForward();
            numCorrect += calculateOutputAccuracy(globalLabels.get(i, 0));
        }
        validationAccuracy = ((double) numCorrect / (double) ((globalFeatures.rows() - (int) (.75 * globalFeatures.rows()))));
    }

    private int calculateOutputAccuracy(double target)
    {
        double highestValue = Double.MIN_VALUE;
        double output = -1;
        for (int i = 0; i < Layer[Layer.length - 1].size; i++)
        {
            if (Layer[Layer.length - 1].output[i] > highestValue)
            {
                highestValue = Layer[Layer.length - 1].output[i];
                output = i;
            }
        }
        if (target == output)
        {
            return 1;
        } else
        {
            return 0;
        }
    }

    private void feedForward()
    {
        for (int i = 0; i < Layer.length - 1; i++)
        {
            //Output Layer
            if (i == Layer.length - 2)
            {
                for (int j = 0; j < Layer[i + 1].size; j++)
                {
                    double outputSum = 0;
                    for (int k = 0; k < Layer[i].size; k++)
                    {
                        outputSum += Layer[i].output[k] * Layer[i].weight[k][j];
                    }
                    Layer[i + 1].output[j] = sigmoid(outputSum);
                }
            } else
            {
                for (int j = 0; j < Layer[i + 1].size; j++)
                {
                    if (j == Layer[i + 1].size - 1)
                    {
                        Layer[i + 1].output[j] = bias;
                    } else
                    {
                        double sum = 0;
                        for (int k = 0; k < Layer[i].size; k++)
                        {
                            sum += Layer[i].output[k] * Layer[i].weight[k][j];
                        }
                        Layer[i + 1].output[j] = sigmoid(sum);
                    }
                }
            }
        }
    }

    private void backprop(double target)
    {

        //Calculate Output Error
        for (int i = 0; i < Layer[Layer.length - 1].size; i++)
        {
            if ((double) i == target)
            {
                Layer[Layer.length - 1].error[i] = ((1 - Layer[Layer.length - 1].output[i]) * (Layer[Layer.length - 1].output[i]) * (1 - Layer[Layer.length - 1].output[i]));
            } else
            {
                Layer[Layer.length - 1].error[i] = ((0 - Layer[Layer.length - 1].output[i]) * (Layer[Layer.length - 1].output[i]) * (1 - Layer[Layer.length - 1].output[i]));
            }
        }

        //Calculate Hidden to Output Layer Weight Change
        for (int i = 0; i < Layer[numberOfLayers - 2].size; i++)
        {
            for (int j = 0; j < Layer[numberOfLayers - 1].size; j++)
            {
                double temp = Layer[numberOfLayers - 2].weightChange[i][j];
                Layer[numberOfLayers - 2].weightChange[i][j] = ((learningRate * (Layer[numberOfLayers - 1].error[j]) * Layer[numberOfLayers - 2].output[i]) + (momentum * temp));
            }
        }

        //Calculate Hidden Layer to Output Error
        for (int i = 0; i < Layer[numberOfLayers - 2].size - 1; i++)
        {
            double sum = 0;
            for (int j = 0; j < Layer[numberOfLayers - 1].size; j++)
            {
                sum += (Layer[numberOfLayers - 1].error[j] * Layer[numberOfLayers - 2].weight[i][j]);
            }
            Layer[numberOfLayers - 2].error[i] = ((Layer[numberOfLayers - 2].output[i]) * (1 - Layer[numberOfLayers - 2].output[i]) * sum);
        }

        //Calculate Hidden Layer to Hidden Layer error
        for (int i = Layer.length - 3; i > 0; i--)
        {
            for (int j = 0; j < Layer[i].size - 1; j++)
            {
                double sum = 0;
                for (int k = 0; k < Layer[i + 1].size - 1; k++)
                {
                    sum += (Layer[i + 1].error[k] * Layer[i].weight[j][k]);
                }
                Layer[i].error[j] = ((Layer[i].output[j]) * (1 - Layer[i].output[j]) * sum);
            }
        }

        //Calculate Hidden Layer to Hidden Layer Weight Change
        for (int i = 1; i < Layer.length - 2; i++)
        {
            for (int j = 0; j < Layer[i].size; j++)
            {
                for (int k = 0; k < Layer[i + 1].size - 1; k++)
                {
                    double temp = Layer[i].weightChange[j][k];
                    Layer[i].weightChange[j][k] = ((learningRate * Layer[i + 1].error[k] * Layer[i].output[j]) + momentum * temp);
                }
            }
        }

        //Calculate input to hidden weight changes
        for (int i = 0; i < Layer[0].size; i++)
        {
            for (int j = 0; j < Layer[1].size - 1; j++)
            {
                double temp = Layer[0].weightChange[i][j];
                Layer[0].weightChange[i][j] = ((learningRate * Layer[1].error[j] * Layer[0].output[i]) + momentum * temp);
            }
        }

        //Change all weights
        for (int i = 0; i < Layer.length - 1; i++)
        {
            for (int j = 0; j < Layer[i].weight.length; j++)
            {
                for (int k = 0; k < Layer[i].weight[0].length; k++)
                {
                    Layer[i].weight[j][k] = ((Layer[i].weight[j][k] + Layer[i].weightChange[j][k]));
                }
            }
        }
    }

    private double sigmoid(double sum)
    {
        return ((1) / (1 + Math.exp(-sum)));
    }

    private void populateInputLayer(int index)
    {
        for (int i = 0; i < inputLayerSize; i++)
        {
            if (i == inputLayerSize - 1)
            {
                Layer[0].output[i] = bias;
            } else
            {
                Layer[0].output[i] = globalFeatures.get(index, i);
            }
        }
    }

    private void populateInputLayer(double[] features)
    {
        for (int i = 0; i <= features.length; i++)
        {
            if (i == features.length)
            {
                Layer[0].output[i] = bias;
            } else
            {
                Layer[0].output[i] = features[i];
            }
        }
    }

    private void intializeWeights()
    {
        for (int i = 0; i < numberOfLayers; i++)
        {
            if (Layer[i].weight != null)
            {
                for (int j = 0; j < Layer[i].weight.length; j++)
                {
                    for (int k = 0; k < Layer[i].weight[j].length; k++)
                    {
                        Layer[i].weight[j][k] = generateRand();
                    }
                }
            }
        }
    }

    private void createLayers()
    {
        Layer = new nLayer[numberOfLayers];
        for (int i = 0; i < numberOfLayers; i++)
        {
            //Input Layer
            if (i == 0)
            {
                //Create input layer and its attributes
                Layer[i] = new nLayer();
                Layer[i].size = inputLayerSize;
                Layer[i].weight = new double[inputLayerSize][hiddenLayerSize - 1];
                Layer[i].weightChange = new double[inputLayerSize][hiddenLayerSize - 1];
                Layer[i].weightStored = new double[inputLayerSize][hiddenLayerSize - 1];
                Layer[i].output = new double[inputLayerSize];
            } else if (i == numberOfLayers - 2)
            {
                //Last hidden layer to output layer
                Layer[i] = new nLayer();
                Layer[i].size = hiddenLayerSize;
                Layer[i].weight = new double[hiddenLayerSize][outputLayerSize];
                Layer[i].weightChange = new double[hiddenLayerSize][outputLayerSize];
                Layer[i].weightStored = new double[hiddenLayerSize][outputLayerSize];
                Layer[i].output = new double[hiddenLayerSize];
                Layer[i].error = new double[hiddenLayerSize - 1];
            }

            //Output Layer
            else if (i == numberOfLayers - 1)
            {
                Layer[i] = new nLayer();
                Layer[i].size = outputLayerSize;
                Layer[i].output = new double[outputLayerSize];
                Layer[i].error = new double[outputLayerSize];
            }
            //Hidden Layers
            else
            {
                Layer[i] = new nLayer();
                Layer[i].size = hiddenLayerSize;
                Layer[i].weight = new double[hiddenLayerSize][hiddenLayerSize - 1];
                Layer[i].weightChange = new double[hiddenLayerSize][hiddenLayerSize - 1];
                Layer[i].weightStored = new double[hiddenLayerSize][hiddenLayerSize - 1];
                Layer[i].output = new double[hiddenLayerSize];
                Layer[i].error = new double[hiddenLayerSize - 1];
            }
        }
    }

    private double generateRand()
    {
        Random generator = new Random();
        int range = generator.nextInt(500 + 1 - 100) - 100;
        return (double) range / 1000;
    }

    public void predict(double[] features, double[] labels)
    {
        populateInputLayer(features);
        feedForward();

        //Get largest output node
        double largestOutput = Double.MIN_VALUE;
        for (int i = 0; i < outputLayerSize; i++)
        {
            if (Layer[Layer.length - 1].output[i] > largestOutput)
            {
                largestOutput = Layer[Layer.length - 1].output[i];
            }
        }

        for (int i = 0; i < outputLayerSize; i++)
        {
            if (Layer[Layer.length - 1].output[i] == largestOutput)
            {
                labels[0] = i;
            }
        }
    }
}
