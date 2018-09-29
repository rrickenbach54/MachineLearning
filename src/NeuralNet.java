import java.util.Arrays;
import java.util.Random;

public class NeuralNet extends SupervisedLearner {
    public NeuralNet(Random rand) {
        super();
    }

    private double [] inputLayer;
    private double [] hiddenLayer;
    private double [] hiddenLayerError;
    private double [] outputLayer;
    private double [] outputLayerError;
    private double [][] inputToHiddenWeights;
    private double [][] inputToHiddenWeightsChange;
    private double [][] hiddenToOutputWeights;
    private double [][] hiddenToOutputWeightsChange;


    private int inputNodes;
    private int hiddenNodes;
    private int outputNodes;
    private double learningRate =0.1;
    private double weightInitialize = -1.0; //Sets weights to be this value set to -1 to randomly generate value.
    private int hiddenLayerScaleSize = 2;
    private int epochs = 1;
    double bias = 1.0;


    public void train(Matrix features, Matrix labels) throws Exception
    {
        this.inputNodes = features.cols() + 1;
        this.hiddenNodes= hiddenLayerScaleSize*inputNodes;//sets the initial hidden layer to be twice the size of input.
        this.outputNodes = labels.valueCount(0);
        createLayers();
        createWeights();
        initializeWeights();
        //printStructure(features);
        travelGradient(features, labels);
    }

    private void travelGradient(Matrix features, Matrix labels)
    {
        for(int i = 0; i<1;i++)
        {
            for(int j =0; j<hiddenLayer.length;j++)
            {
                calculateHiddenOutput(features,j);
            }
            for(int k =0; k<outputLayer.length;k++)
            {
                calculateOutputSum(k);
            }
        }
    }

    private void printStructure(Matrix features)
    {
        //Print input layer
        for(int i=0; i<inputLayer.length;i++)
        {
            if(i == inputLayer.length -1)
            {
                System.out.print("  [" + i +"]   Value: " + bias);
            }
            else {
                System.out.print("  [" + i + "]   Value: " + features.get(0, i));
            }
        }
        System.out.println();

        //Print weights be sent from input to hidden layer
        for(int i =0;i<inputLayer.length;i++)
        {
            for(int j=0;j<hiddenLayer.length;j++)
            {
                System.out.print("[" + i + "][" + j + "] Value: " + inputToHiddenWeights[i][j] + " ");
            }
            System.out.print("  |   ");
        }
        System.out.println();
        System.out.println();

        //Print hidden layer
        for(int i=0; i<hiddenLayer.length;i++)
        {
            if(i == hiddenLayer.length -1)
            {
                System.out.print("  [" + i +"]   Value: " + bias);
            }
            else
            {
                System.out.print("  [" + i +"]   Value: " + calculateHiddenOutput(features, i));
            }

        }
        System.out.println();

        //Print weights being sent from hidden to output layer
        for(int i =0;i<hiddenLayer.length;i++)
        {
            for(int j=0;j<outputLayer.length;j++)
            {
                System.out.print(" [" + i + "][" + j + "] Value: " + hiddenToOutputWeights[i][j]);
            }
            System.out.print("  |   ");
        }
        System.out.println();
        System.out.println();

        //Print output layer
        for(int i=0; i<outputLayer.length;i++)
        {
            System.out.print("  [" + i +"]  Value: " + calculateOutputSum(i));
        }
        System.out.println();
        System.out.println();

    }

    private double calculateOutputSum(int index)
    {
        double sum = 0;
        for(int i =0;i<hiddenLayer.length;i++)
        {
            sum += hiddenLayer[i]*hiddenToOutputWeights[i][index];
        }
        double output = ((1)/(1+Math.exp(-sum)));//Sigmoid function
        outputLayer[index]=output;
        return sum;
    }

    private double calculateHiddenOutput(Matrix features, int index)
    {
        populateInputArray(features, 0);
        double sum = 0;
        for(int i =0;i<inputLayer.length;i++)
        {
            //[0]*[0][0] + [1]*[1][0]
            sum += inputLayer[i]*inputToHiddenWeights[i][index];
        }
        double output = ((1)/(1+Math.exp(-sum))); //Sigmoid function
        hiddenLayer[index] = output;
        return output;
    }

    private void populateInputArray(Matrix features, int index)
    {
        for (int i = 0; i < inputLayer.length;i++)
        {
            if(i == inputLayer.length -1)
            {
                inputLayer[i] = bias;
            }
            else {
                inputLayer[i] = features.get(index, i);
            }
        }
    }


    private void initializeWeights()
    {
        if (weightInitialize > 0)
        {
            //Initialize input to hidden layer weights to be a specified value
            for(int i = 0; i<inputLayer.length;i++)
            {
                for(int j = 0; j<hiddenLayer.length;j++)
                {
                    inputToHiddenWeights[i][j] = weightInitialize;
                }
            }

            //Initialize hidden to output layer weights to be a specified value if set.
            for(int i = 0; i<hiddenLayer.length;i++)
            {
                for(int j = 0; j<outputLayer.length;j++)
                {
                    hiddenToOutputWeights[i][j] = weightInitialize;
                }
            }
        }
        else
        {
            //Initialize input to hidden layer weights randomly
            for(int i = 0; i<inputLayer.length;i++)
            {
                for(int j = 0; j<hiddenLayer.length;j++)
                {
                    inputToHiddenWeights[i][j] = generateRand();
                }
            }

            //Initialize hidden to output layer weights randomly
            for(int i = 0; i<hiddenLayer.length;i++)
            {
                for(int j = 0; j<outputLayer.length;j++)
                {
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

    private void createWeights()
    {
        inputToHiddenWeights = new double [inputLayer.length][hiddenLayer.length];
        inputToHiddenWeightsChange = new double[inputLayer.length][hiddenLayer.length];//Temp Array for holding the changes in weights.
        hiddenToOutputWeights = new double[hiddenLayer.length][outputLayer.length];
        hiddenToOutputWeightsChange = new double [hiddenLayer.length][outputLayer.length];//Temp array for holding the changes in weights from hidden to output.

    }

    private void createLayers()
    {
        //Initializes the arrays to be the correct size based on the inputs and specifications
        inputLayer = new double[inputNodes];
        hiddenLayer = new double [hiddenNodes];
        hiddenLayerError = new double [hiddenNodes];//Array for holding the errors of hidden nodes
        outputLayer = new double [outputNodes];
        outputLayerError = new double[outputNodes];//Array for holding the errors of output nodes
    }

    public void predict(double[] features, double[] labels) throws Exception {

    }


}
