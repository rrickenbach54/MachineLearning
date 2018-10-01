import java.util.Arrays;
import java.util.Random;

public class NeuralNet extends SupervisedLearner {
    public NeuralNet(Random rand) {
        super();
    }

    private double[] inputLayer;
    private double[] hiddenLayer;
    private double[] hiddenLayerError;
    private double[] outputLayer;
    private double[] outputLayerError;
    private double[][] inputToHiddenWeights;
    private double[][] inputToHiddenWeightsChange;
    private double[][] hiddenToOutputWeights;
    private double[][] hiddenToOutputWeightsChange;


    private int inputNodes;
    private int hiddenNodes;
    private int outputNodes;
    private double learningRate = 1;
    private double weightInitialize = 1.0; //Sets weights to be this value set to -1 to randomly generate value.
    private int hiddenLayerScaleSize = 2;
    private int epochs = 1;
    double bias = 1.0;


    public void train(Matrix features, Matrix labels) throws Exception {
        this.inputNodes = features.cols() + 1;
        this.hiddenNodes = hiddenLayerScaleSize * inputNodes;//sets the initial hidden layer to be twice the size of input.
        this.outputNodes = labels.valueCount(0);
        createLayers();
        createWeights();
        initializeWeights();
        //printStructure(features);
        for (int i = 0; i < 10000; i++) {
            travelGradient(features, labels);
            //printStructure(features);
            features.shuffle(new Random(), labels);
        }

    }


    private void travelGradient(Matrix features, Matrix labels) {
        for (int i = 0; i < features.rows(); i++) {
            populateInputArray(features,i);
            for (int j = 0; j < hiddenLayer.length; j++) {
                if (j == hiddenLayer.length - 1) {
                    hiddenLayer[j] = bias;
                } else {
                    calculateHiddenOutput(features, j);
                }
            }
            for (int k = 0; k < outputLayer.length; k++) {
                calculateOutput(k);
            }
            backprop(features, labels, i);
        }
    }

    private void backprop(Matrix features, Matrix labels, int index) {

        //Calculate Output Error
        double target = labels.get(index, 0);
        for (int i = 0; i < outputLayer.length; i++) {
            if (outputLayer.length == 1) {
                outputLayerError[i] = ((target - outputLayer[i]) * outputLayer[i] * (1 - outputLayer[i]));
                //System.out.println("Output [" + i + "] Error: " + outputLayerError[i]);
                //System.out.println("Output [" + i + "] Error: " + outputLayerError[i]);
            } else {
                if ((double) (i) == target) {
                    outputLayerError[i] = ((1 - outputLayer[i]) * ((outputLayer[i]) * (1 - outputLayer[i])));
                } else {
                    outputLayerError[i] = (0 - outputLayer[i]) * (outputLayer[i]) * (1 - outputLayer[i]);
                }
                //System.out.println("Output [" + i + "] Error: " + outputLayerError[i]);
            }
        }

        //Calculate Hidden to Output Layer Weight Change
        for (int i = 0; i < hiddenLayer.length; i++) {
            for (int j = 0; j < outputLayer.length; j++) {
                hiddenToOutputWeightsChange[i][j] = (learningRate * (outputLayerError[j]) * hiddenLayer[i]);
                //System.out.println("Weight Change [" +i+"]["+j+"]: " + hiddenToOutputWeightsChange[i][j]);
            }
        }

        //Calculate Hidden Layer Error
        for (int i = 0; i < hiddenLayer.length - 1; i++) {
            double sum = 0;
            for (int j = 0; j < outputLayer.length; j++) {
                sum += (outputLayerError[j] * hiddenToOutputWeights[i][j]);
            }
            hiddenLayerError[i] = ((hiddenLayer[i]) * (1 - hiddenLayer[i]) * sum);
            //System.out.println("Hidden Layer: " + i +" Error: " + hiddenLayerError[i]);
        }

        //Calculate Input to Hidden Layer Weight Change
        for (int i = 0; i < inputLayer.length; i++) {
            for (int j = 0; j < hiddenLayer.length - 1; j++) {
                inputToHiddenWeightsChange[i][j] = (learningRate * hiddenLayerError[j] * inputLayer[i]);
                //System.out.println("Changing weight [" +i +"][" +j +"] by: " + (learningRate*hiddenLayerError[j]*inputLayer[i]));
            }
        }

        //Change the weights from hidden to output
        for (int i = 0; i < hiddenLayer.length; i++) {
            for (int j = 0; j < outputLayer.length; j++) {
                hiddenToOutputWeights[i][j] = ((hiddenToOutputWeights[i][j]) + (hiddenToOutputWeightsChange[i][j]));
                hiddenToOutputWeightsChange[i][j] = 0.0; //Reset the temp back to 0 after using.
            }
        }

        //Change the weights from input to hidden
        for (int i = 0; i < inputLayer.length; i++) {
            for (int j = 0; j < hiddenLayer.length - 1; j++) {
                inputToHiddenWeights[i][j] = ((inputToHiddenWeights[i][j]) + (inputToHiddenWeightsChange[i][j]));
                inputToHiddenWeightsChange[i][j] = 0.0; //Reset the temp back to 0 after using.
            }
        }
    }

    private void printStructure(Matrix features) {
        //Print input layer
        for (int i = 0; i < inputLayer.length; i++) {
            if (i == inputLayer.length - 1) {
                System.out.print("  [" + i + "]   Value: " + bias);
            } else {
                System.out.print("  [" + i + "]   Value: " + features.get(0, i));
            }
        }
        System.out.println();

        //Print weights be sent from input to hidden layer
        for (int i = 0; i < inputLayer.length; i++) {
            for (int j = 0; j < hiddenLayer.length - 1; j++) {
                System.out.print("[" + i + "][" + j + "] Value: " + inputToHiddenWeights[i][j] + " ");
            }
            System.out.print("  |   ");
        }
        System.out.println();
        System.out.println();

        //Print hidden layer
        for (int i = 0; i < hiddenLayer.length; i++) {
            if (i == hiddenLayer.length - 1) {
                System.out.print("  [" + i + "]   Value: " + bias);
            } else {
                System.out.print("  [" + i + "]   Value: " + calculateHiddenOutput(features, i));
            }

        }
        System.out.println();

        //Print weights being sent from hidden to output layer
        for (int i = 0; i < hiddenLayer.length; i++) {
            for (int j = 0; j < outputLayer.length; j++) {
                System.out.print(" [" + i + "][" + j + "] Value: " + hiddenToOutputWeights[i][j]);
            }
            System.out.print("  |   ");
        }
        System.out.println();
        System.out.println();

        //Print output layer
        for (int i = 0; i < outputLayer.length; i++) {
            System.out.print("  [" + i + "]  Value: " + calculateOutput(i));
        }
        System.out.println();
        System.out.println();

    }

    private double calculateOutput(int index) {
        double sum = 0;
        for (int i = 0; i < hiddenLayer.length; i++) {
            sum += hiddenLayer[i] * hiddenToOutputWeights[i][index];
        }
        double output = ((1) / (1 + Math.exp(-sum)));//Sigmoid function
        outputLayer[index] = output;
        return sum;
    }

    private double calculateHiddenOutput(Matrix features, int index) {
        double sum = 0;
        for (int i = 0; i < inputLayer.length; i++) {
            //[0]*[0][0] + [1]*[1][0]
            sum += inputLayer[i] * inputToHiddenWeights[i][index];
        }
        double output = ((1) / (1 + Math.exp(-sum))); //Sigmoid function
        hiddenLayer[index] = output;
        return output;

    }

    private void populateInputArray(Matrix features, int index) {
        for (int i = 0; i < inputLayer.length; i++) {
            if (i == inputLayer.length - 1) {
                inputLayer[i] = bias;
            } else {
                inputLayer[i] = features.get(index, i);
            }
        }
    }


    private void initializeWeights() {
        if (weightInitialize > 0) {
            //Initialize input to hidden layer weights to be a specified value
            for (int i = 0; i < inputLayer.length; i++) {
                for (int j = 0; j < hiddenLayer.length - 1; j++) {
                    inputToHiddenWeights[i][j] = weightInitialize;
                }
            }

            //Initialize hidden to output layer weights to be a specified value if set.
            for (int i = 0; i < hiddenLayer.length; i++) {
                for (int j = 0; j < outputLayer.length; j++) {
                    hiddenToOutputWeights[i][j] = weightInitialize;
                }
            }
        } else {
            //Initialize input to hidden layer weights randomly
            for (int i = 0; i < inputLayer.length; i++) {
                for (int j = 0; j < hiddenLayer.length-1; j++) {
                    inputToHiddenWeights[i][j] = generateRand();
                }
            }

            //Initialize hidden to output layer weights randomly
            for (int i = 0; i < hiddenLayer.length; i++) {
                for (int j = 0; j < outputLayer.length; j++) {
                    hiddenToOutputWeights[i][j] = generateRand();
                }
            }
        }
    }

    //Create a number between -0.5 and 0.5 used when randomly setting weights we want it to be close to 0 for sigmoid function.
    private double generateRand() {
        Random generator = new Random();
        int range = generator.nextInt(500 + 1 - 100) - 100;
        double rand = (double) range / 1000;
        return rand;
    }

    private void createWeights() {
        inputToHiddenWeights = new double[inputLayer.length][hiddenLayer.length - 1];
        inputToHiddenWeightsChange = new double[inputLayer.length][hiddenLayer.length - 1];//Temp Array for holding the changes in weights.
        hiddenToOutputWeights = new double[hiddenLayer.length][outputLayer.length];
        hiddenToOutputWeightsChange = new double[hiddenLayer.length][outputLayer.length];//Temp array for holding the changes in weights from hidden to output.

    }

    private void createLayers() {
        //Initializes the arrays to be the correct size based on the inputs and specifications
        inputLayer = new double[inputNodes];
        hiddenLayer = new double[hiddenNodes];
        hiddenLayerError = new double[hiddenNodes - 1];//Array for holding the errors of hidden nodes
        outputLayer = new double[outputNodes];
        outputLayerError = new double[outputNodes];//Array for holding the errors of output nodes
    }

    public void predict(double[] features, double[] labels) throws Exception {
        //Populate the input layer with bias from features array
        for (int i = 0; i < features.length + 1; i++) {
            if (i == features.length) {
                inputLayer[i] = bias;
            } else {
                inputLayer[i] = features[i];
            }
        }

        //Calculate Hidden Output from features array
        for (int i = 0; i < hiddenLayer.length - 1; i++) {
            double hiddenSum = 0.0;
            for (int j = 0; j < inputLayer.length; j++) {
                if (j == inputLayer.length - 1) {
                    hiddenSum += inputLayer[j] * inputToHiddenWeights[j][i];
                    double hiddenOutput = ((1) / (1 + Math.exp(-hiddenSum))); //Sigmoid function
                    hiddenLayer[i] = hiddenOutput;
                } else {
                    hiddenSum += inputLayer[j] * inputToHiddenWeights[j][i];
                }
            }
        }

        //Calculate Output Nodes
        for (int i = 0; i < outputLayer.length; i++) {
            double outputSum = 0.0;
            for (int j = 0; j < hiddenLayer.length; j++) {
                if (j == hiddenLayer.length - 1) {
                    outputSum += hiddenLayer[j] * hiddenToOutputWeights[j][i];
                    double output = ((1) / (1 + Math.exp(-outputSum))); //Sigmoid function
                    outputLayer[i] = output;
                } else {
                    outputSum += hiddenLayer[j] * hiddenToOutputWeights[j][i];
                }
            }
        }

        //Give final output
        double largestValue = -10.0;
        for (int i = 0; i < outputLayer.length; i++) {

            if (outputLayer[i] > largestValue) {
                largestValue = outputLayer[i];
            }
        }
        for (int i = 0; i < outputLayer.length; i++) {
            if (outputLayer[i] == largestValue) {
                labels[0] = i;
            }
        }

    }
}
