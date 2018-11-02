public class DecisionTree extends SupervisedLearner
{
    private int labelCount [];
    private int featureCount [];
    private int featureCountNum=0;
    private int featureLabel[];
    private double gainArray [];
    
    @Override
    public void train(Matrix features, Matrix labels) throws Exception
    {
        createGainArray(features);
        populateLabelCount(labels);
        //Calculate Info
        double infoS = 0;

        for(int i=0; i<labelCount.length;i++)
            infoS += -1 * ((labelCount[i] / (double) labels.rows()) * log2((labelCount[i] / (double) labels.rows())));
        System.out.println("InfoS: " + infoS);

        //Calculate Gain
        for(int i =0; i<features.cols();i++)
        {
            double infoAttr = 0;
            populateFeatureCount(features,i);
            for(int j=0;j<featureCount.length;j++)
            {
                populateFeatureLabel(features,labels,i,j);
                double inside=0;
                for(int k=0;k<featureLabel.length;k++)
                {
                    if(Double.isNaN((((double)featureLabel[k]/featureCount[j]))*(log2((double)featureLabel[k]/featureCount[j]))))
                    {
                    }
                    else
                    {
                        inside -= ((((double) featureLabel[k] / featureCount[j])) * (log2((double) featureLabel[k] / featureCount[j])));
                    }
                }
                infoAttr += (((double)featureCount[j]/featureCountNum)*inside);
            }
            double gain = infoS - infoAttr;
            System.out.println("Gain[" + i + "]: " + gain);
            gainArray[i] = gain;
        }

        //Decide which feature to split on
        double max =0;
        int index= -1;
        for(int i=0;i<gainArray.length;i++)
        {
            if(gainArray[i] >max)
            {
                max = gainArray[i];
                index = i;
            }
        }
        System.out.println("Split on feature: " + index);
    }

    private void populateFeatureLabel(Matrix features, Matrix labels, int index,int value)
    {
        featureLabel = new int[labels.valueCount(0)];
        for(int i=0;i<features.rows();i++)
        {
            if(features.get(i,index)==value)
            {
                featureLabel[(int)labels.get(i,0)]++;
            }
        }
    }


    private void populateFeatureCount(Matrix features,int i)
    {
        featureCount = new int[features.valueCount(i)];
        featureCountNum=0;
        for(int j =0; j<features.rows();j++)
        {
            featureCount[(int)features.get(j,i)]++;
            featureCountNum++;
        }
    }



    private double log2(double v)
    {
        return Math.log(v)/Math.log(2.0);
    }

    private void populateLabelCount(Matrix labels)
    {
        labelCount = new int[labels.valueCount(0)];
        for(int i =0;i<labels.rows();i++)
        {
            labelCount[(int)labels.get(i,0)]++;
        }
    }

    private void createGainArray(Matrix features)
    {
        gainArray = new double[features.cols()];
    }

    @Override
    public void predict(double[] features, double[] labels) throws Exception
    {

    }
}
