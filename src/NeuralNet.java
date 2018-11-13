import java.util.Random;

public class NeuralNet extends SupervisedLearner
{
    public NeuralNet(Random rand)
    {
        super();
    }

    private double[] inputLayer;
    private double[] hiddenLayer;
    private double[] hiddenLayerError;
    private double[] outputLayer;
    private double[] outputLayerError;
    private double[][] inputToHiddenWeights;
    private double[][] inputToHiddenWeightsStored;
    private double[][] inputToHiddenWeightsChange;
    private double[][] hiddenToOutputWeights;
    private double[][] hiddenToOutputWeightsStored;
    private double[][] hiddenToOutputWeightsChange;

    private int inputNodes;
    private int hiddenNodes;
    private int outputNodes;
    private int epochs = 1;
    private double bias = 1.0;
    private boolean validate = true;
    private boolean continueTraining = true;
    private double validationAccuracy = 0;
    private double previousValidationAccuracy = 0;
    private double maxAccuracy = 0;
    private int runsWithoutImprovement = 0;



    public void train(Matrix features, Matrix labels)
    {
        this.inputNodes = features.cols() + 1;
        double hiddenLayerScaleSize = 2;
        this.hiddenNodes = (int)(hiddenLayerScaleSize * inputNodes);//sets the initial hidden layer to be twice the size of input.
        this.outputNodes = labels.valueCount(0);
        createLayers();
        createWeights();
        initializeWeights();
        //printStructure(features);

        //Max Epochs the program can run, rarely hits this point but is there just in case.
        int maxEpochs = 10000;
        while (epochs < maxEpochs && continueTraining)
        {

            feedForward(features, labels);
            //printStructure(features);
            calculateAccuracy(features, labels);

            //Checks the validation set accuracy and contains the criteria for stop training
            if (validate)
            {
                calculateValidationAccuracy(features, labels);
                if ((validationAccuracy - previousValidationAccuracy) > 0.01)
                {
                    previousValidationAccuracy = validationAccuracy;
                    runsWithoutImprovement = 0;
                    if (validationAccuracy > maxAccuracy)
                    {
                        //Store the best weights
                        storeWeights();
                        maxAccuracy = validationAccuracy;
                        if(validationAccuracy ==100)
                        {
                            continueTraining=false;
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
                //System.out.println("Validation Set Accuracy " + validationAccuracy);
            }
            features.shuffle(new Random(), labels);
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


    //Function to restore weights
    private void restoreWeights()
    {
        //Restore Input to Hidden Weights
        for (int i = 0; i < inputLayer.length; i++)
        {
            if (hiddenLayer.length - 1 >= 0)
                System.arraycopy(inputToHiddenWeightsStored[i], 0, inputToHiddenWeights[i], 0, hiddenLayer.length - 1);
        }

        //Restore hidden to output layer weights
        for (int i = 0; i < hiddenLayer.length; i++)
        {
            System.arraycopy(hiddenToOutputWeightsStored[i], 0, hiddenToOutputWeights[i], 0, outputLayer.length);
        }
    }

    //Store best accuracy weights
    private void storeWeights()
    {
        //Store Input to Hidden Weights
        for (int i = 0; i < inputLayer.length; i++)
        {
            if (hiddenLayer.length - 1 >= 0)
                System.arraycopy(inputToHiddenWeights[i], 0, inputToHiddenWeightsStored[i], 0, hiddenLayer.length - 1);
        }

        //Store hidden to output layer weights
        for (int i = 0; i < hiddenLayer.length; i++)
        {
            System.arraycopy(hiddenToOutputWeights[i], 0, hiddenToOutputWeightsStored[i], 0, outputLayer.length);
        }
    }


    //Feed forward function
    //Populates the neural net then calls back prop for updating
    //If using validation set it just uses the first 75% of features
    private void feedForward(Matrix features, Matrix labels)
    {
        if (validate)
        {
            for (int i = 0; i < (int) (.75 * features.rows()+1); i++)
            {
                populateInputArray(features, i);
                for (int j = 0; j < hiddenLayer.length; j++)
                {
                    if (j == hiddenLayer.length - 1)
                    {
                        hiddenLayer[j] = bias;
                    } else
                    {
                        calculateHiddenOutput(j);
                    }
                }
                for (int k = 0; k < outputLayer.length; k++)
                {
                    calculateOutput(k);
                }
                backprop(labels, i);
            }
        } else
        {
            for (int i = 0; i < features.rows(); i++)
            {
                populateInputArray(features, i);
                for (int j = 0; j < hiddenLayer.length; j++)
                {
                    if (j == hiddenLayer.length - 1)
                    {
                        hiddenLayer[j] = bias;
                    } else
                    {
                        calculateHiddenOutput(j);
                    }
                }
                for (int k = 0; k < outputLayer.length; k++)
                {
                    calculateOutput(k);
                }
                backprop(labels, i);
            }
        }
    }

    //Method used for calculating training accuracy without validation
    private void calculateAccuracy(Matrix features, Matrix labels)
    {
        int numCorrect =0;
        for (int i = 0; i < features.rows(); i++)
        {
            populateInputArray(features, i);
            for (int j = 0; j < hiddenLayer.length; j++)
            {
                if (j == hiddenLayer.length - 1)
                {
                    hiddenLayer[j] = bias;
                } else
                {
                    calculateHiddenOutput(j);
                }
            }
            for (int k = 0; k < outputLayer.length; k++)
            {
                calculateOutput(k);
            }
            numCorrect += calculateOutputAccuracy(labels, i);
        }
        //System.out.print(1-((double)numCorrect/(double)features.rows())+",");
        //System.out.println("Epoch: " + epochs + " Accuracy: " + numCorrect + " Out of: " + features.rows() + " Percent: " + ((double)numCorrect/(double)features.rows()*100));
    }

    //Calculates validation accuracy
    private void calculateValidationAccuracy(Matrix features, Matrix labels)
    {
        int numCorrect = 0;
        //System.out.println("Validation Starts at " + (int) (.75 * features.rows()));
        for (int i = (int) (.75 * features.rows()); i < features.rows(); i++)
        {
            populateInputArray(features, i);
            for (int j = 0; j < hiddenLayer.length; j++)
            {
                if (j == hiddenLayer.length - 1)
                {
                    hiddenLayer[j] = bias;
                } else
                {
                    calculateHiddenOutput(j);
                }
            }
            for (int k = 0; k < outputLayer.length; k++)
            {
                calculateOutput(k);
            }
            numCorrect += calculateOutputAccuracy(labels, i);
        }
        //System.out.print(1-((double)numCorrect/(double)features.rows())+",");
        //System.out.println("Validation Set Size: " + (double) ((features.rows() - (int) (.75 * features.rows()))) + " Number Correct " + numCorrect);
        validationAccuracy = ((double) numCorrect / (double) ((features.rows() - (int) (.75 * features.rows()))));
        //System.out.print(1-validationAccuracy+",");
    }

    private int calculateOutputAccuracy(Matrix labels, int index)
    {
        double highestValue = -10;
        double target = labels.get(index, 0);
        double output = -1;
        for (int i = 0; i < outputLayer.length; i++)
        {
            if (outputLayer[i] > highestValue)
            {
                highestValue = outputLayer[i];
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


    //Backpropagation function
    private void backprop(Matrix labels, int index)
    {

        //Calculate Output Error
        double target = labels.get(index, 0);
        for (int i = 0; i < outputLayer.length; i++)
        {
            if (outputLayer.length == 1)
            {
                outputLayerError[i] = ((target - outputLayer[i]) * outputLayer[i] * (1 - outputLayer[i]));
                //System.out.println("Output [" + i + "] Error: " + outputLayerError[i]);
                //System.out.println("Output [" + i + "] Error: " + outputLayerError[i]);
            } else
            {
                if ((double) (i) == target)
                {
                    outputLayerError[i] = ((1 - outputLayer[i]) * ((outputLayer[i]) * (1 - outputLayer[i])));
                } else
                {
                    outputLayerError[i] = (0 - outputLayer[i]) * (outputLayer[i]) * (1 - outputLayer[i]);
                }
                //System.out.println("Output [" + i + "] Error: " + outputLayerError[i]);
            }
        }

        //Calculate Hidden to Output Layer Weight Change
        double momentum = 0.3;

        //Learning rate set to 0.05 was the best functionality in testing
        double learningRate = 0.05;
        for (int i = 0; i < hiddenLayer.length; i++)
        {
            for (int j = 0; j < outputLayer.length; j++)
            {

                double temp = hiddenToOutputWeightsChange[i][j];
                hiddenToOutputWeightsChange[i][j] = ((learningRate * (outputLayerError[j]) * hiddenLayer[i]) + momentum *temp); //MOMENTUM
                //System.out.println("Weight Change [" +i+"]["+j+"]: " + hiddenToOutputWeightsChange[i][j]);
            }
        }

        //Calculate Hidden Layer Error
        for (int i = 0; i < hiddenLayer.length - 1; i++)
        {
            double sum = 0;
            for (int j = 0; j < outputLayer.length; j++)
            {
                sum += (outputLayerError[j] * hiddenToOutputWeights[i][j]);
            }
            hiddenLayerError[i] = ((hiddenLayer[i]) * (1 - hiddenLayer[i]) * sum);
            //System.out.println("Hidden Layer: " + i +" Error: " + hiddenLayerError[i]);
        }

        //Calculate Input to Hidden Layer Weight Change
        for (int i = 0; i < inputLayer.length; i++)
        {
            for (int j = 0; j < hiddenLayer.length - 1; j++)
            {
                double temp = inputToHiddenWeightsChange[i][j];
                inputToHiddenWeightsChange[i][j] = ((learningRate * hiddenLayerError[j] * inputLayer[i])+ momentum *temp); //MOMENTUM
                //System.out.println("Changing weight [" +i +"][" +j +"] by: " + (learningRate*hiddenLayerError[j]*inputLayer[i]));
            }
        }

        //Change the weights from hidden to output
        for (int i = 0; i < hiddenLayer.length; i++)
        {
            for (int j = 0; j < outputLayer.length; j++)
            {
                hiddenToOutputWeights[i][j] = ((hiddenToOutputWeights[i][j]) + (hiddenToOutputWeightsChange[i][j]));
                //hiddenToOutputWeightsChange[i][j] = 0.0; //Reset the temp back to 0 after using.
            }
        }

        //Change the weights from input to hidden
        for (int i = 0; i < inputLayer.length; i++)
        {
            for (int j = 0; j < hiddenLayer.length - 1; j++)
            {
                inputToHiddenWeights[i][j] = ((inputToHiddenWeights[i][j]) + (inputToHiddenWeightsChange[i][j]));
                //inputToHiddenWeightsChange[i][j] = 0.0; //Reset the temp back to 0 after using.
            }
        }
    }

    //Method that was used in the begging to see how all the network is structured left in can be useful for debugging.
//    private void printStructure(Matrix features) {
//        //Print input layer
//        for (int i = 0; i < inputLayer.length; i++) {
//            if (i == inputLayer.length - 1) {
//                System.out.print("  [" + i + "]   Value: " + bias);
//            } else {
//                System.out.print("  [" + i + "]   Value: " + features.get(0, i));
//            }
//        }
//        System.out.println();
//
//        //Print weights be sent from input to hidden layer
//        for (int i = 0; i < inputLayer.length; i++) {
//            for (int j = 0; j < hiddenLayer.length - 1; j++) {
//                System.out.print("[" + i + "][" + j + "] Value: " + inputToHiddenWeights[i][j] + " ");
//            }
//            System.out.print("  |   ");
//        }
//        System.out.println();
//        System.out.println();
//
//        //Print hidden layer
//        for (int i = 0; i < hiddenLayer.length; i++) {
//            if (i == hiddenLayer.length - 1) {
//                System.out.print("  [" + i + "]   Value: " + bias);
//            } else {
//                System.out.print("  [" + i + "]   Value: " + calculateHiddenOutput(features, i));
//            }
//
//        }
//        System.out.println();
//
//        //Print weights being sent from hidden to output layer
//        for (int i = 0; i < hiddenLayer.length; i++) {
//            for (int j = 0; j < outputLayer.length; j++) {
//                System.out.print(" [" + i + "][" + j + "] Value: " + hiddenToOutputWeights[i][j]);
//            }
//            System.out.print("  |   ");
//        }
//        System.out.println();
//        System.out.println();
//
//        //Print output layer
//        for (int i = 0; i < outputLayer.length; i++) {
//            System.out.print("  [" + i + "]  Value: " + calculateOutput(i));
//        }
//        System.out.println();
//        System.out.println();
//
//    }

    //Utility method used in feed forward to calculate output nodes
    private void calculateOutput(int index)
    {
        double sum = 0;
        for (int i = 0; i < hiddenLayer.length; i++)
        {
            sum += hiddenLayer[i] * hiddenToOutputWeights[i][index];
        }
        double output = ((1) / (1 + Math.exp(-sum)));//Sigmoid function
        outputLayer[index] = output;
        //return output;
    }

    //Utility method used in feed forward to calculate hidden nodes
    private void calculateHiddenOutput(int index)
    {
        double sum = 0;
        for (int i = 0; i < inputLayer.length; i++)
        {
            //[0]*[0][0] + [1]*[1][0]
            sum += inputLayer[i] * inputToHiddenWeights[i][index];
        }
        double output = ((1) / (1 + Math.exp(-sum))); //Sigmoid function
        hiddenLayer[index] = output;
        //return output;

    }

    //Utility method to feed in feature set into input array
    private void populateInputArray(Matrix features, int index)
    {
        for (int i = 0; i < inputLayer.length; i++)
        {
            if (i == inputLayer.length - 1)
            {
                inputLayer[i] = bias;
            } else
            {
                inputLayer[i] = features.get(index, i);
            }
        }
    }


    //Initialize weights if not set to 0 or greater then randomly set values between -0.5 & 0.5
    private void initializeWeights()
    {
        //Sets weights to be this value set to -1 to randomly generate value.
        //Set to 0 or above to set all weights to that number
        double weightInitialize = -1;
        if (weightInitialize >= 0)
        {
            //Initialize input to hidden layer weights to be a specified value
            for (int i = 0; i < inputLayer.length; i++)
            {
                for (int j = 0; j < hiddenLayer.length - 1; j++)
                {
                    inputToHiddenWeights[i][j] = weightInitialize;
                }
            }

            //Initialize hidden to output layer weights to be a specified value if set.
            for (int i = 0; i < hiddenLayer.length; i++)
            {
                for (int j = 0; j < outputLayer.length; j++)
                {
                    hiddenToOutputWeights[i][j] = weightInitialize;
                }
            }
        } else
        {
            //Initialize input to hidden layer weights randomly
            for (int i = 0; i < inputLayer.length; i++)
            {
                for (int j = 0; j < hiddenLayer.length - 1; j++)
                {
                    inputToHiddenWeights[i][j] = generateRand();
                }
            }

            //Initialize hidden to output layer weights randomly
            for (int i = 0; i < hiddenLayer.length; i++)
            {
                for (int j = 0; j < outputLayer.length; j++)
                {
                    hiddenToOutputWeights[i][j] = generateRand();
                }
            }
        }
    }

    //Create a number between -0.5 and 0.5 used when randomly setting weights we want it to be close to 0 for sigmoid function.
    private double generateRand()
    {
        Random generator = new Random();
        int range = generator.nextInt(500 + 1 - 100) - 100;
        return (double) range / 1000;
    }

    //Creates the weight arrays to specified sizes
    private void createWeights()
    {
        inputToHiddenWeights = new double[inputLayer.length][hiddenLayer.length - 1];
        inputToHiddenWeightsStored = new double[inputLayer.length][hiddenLayer.length - 1];
        inputToHiddenWeightsChange = new double[inputLayer.length][hiddenLayer.length - 1];//Temp Array for holding the changes in weights.
        hiddenToOutputWeights = new double[hiddenLayer.length][outputLayer.length];
        hiddenToOutputWeightsStored = new double[hiddenLayer.length][outputLayer.length];
        hiddenToOutputWeightsChange = new double[hiddenLayer.length][outputLayer.length];//Temp array for holding the changes in weights from hidden to output.

    }

    //Creates the nodes of specified sizes
    private void createLayers()
    {
        //Initializes the arrays to be the correct size based on the inputs and specifications
        inputLayer = new double[inputNodes];
        hiddenLayer = new double[hiddenNodes];
        hiddenLayerError = new double[hiddenNodes - 1];//Array for holding the errors of hidden nodes
        outputLayer = new double[outputNodes];
        outputLayerError = new double[outputNodes];//Array for holding the errors of output nodes
    }

    //Predict function called by supervised learner.
    public void predict(double[] features, double[] labels)
    {
        //Populate the input layer with bias from features array
        for (int i = 0; i < features.length + 1; i++)
        {
            if (i == features.length)
            {
                inputLayer[i] = bias;
            } else
            {
                inputLayer[i] = features[i];
            }
        }

        //Calculate Hidden Output from features array
        for (int i = 0; i < hiddenLayer.length - 1; i++)
        {
            double hiddenSum = 0.0;
            calculateNodes(i, hiddenSum, inputLayer, inputToHiddenWeights, hiddenLayer);
        }

        //Calculate Output Nodes
        for (int i = 0; i < outputLayer.length; i++)
        {
            double outputSum = 0.0;
            calculateNodes(i, outputSum, hiddenLayer, hiddenToOutputWeights, outputLayer);
        }

        //Give final output
        double largestValue = -10.0;
        for (double anOutputLayer : outputLayer)
        {

            if (anOutputLayer > largestValue)
            {
                largestValue = anOutputLayer;
            }
        }
        for (int i = 0; i < outputLayer.length; i++)
        {
            if (outputLayer[i] == largestValue)
            {
                labels[0] = i;
            }
        }

    }

    //Utility method used in predict to do feed forward.
    private void calculateNodes(int i, double hiddenSum, double[] inputLayer, double[][] inputToHiddenWeights, double[] hiddenLayer)
    {
        for (int j = 0; j < inputLayer.length; j++)
        {
            if (j == inputLayer.length - 1)
            {
                hiddenSum += inputLayer[j] * inputToHiddenWeights[j][i];
                double hiddenOutput = ((1) / (1 + Math.exp(-hiddenSum))); //Sigmoid function
                hiddenLayer[i] = hiddenOutput;
            } else
            {
                hiddenSum += inputLayer[j] * inputToHiddenWeights[j][i];
            }
        }
    }
}
