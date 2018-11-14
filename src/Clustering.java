import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class Clustering
{
    int kMeans = 5;
    double [][] distances;
    int [] clusters;
    Matrix centroids;
    HashMap<Integer, List<Integer>> map = new HashMap<Integer, List<Integer>>();


    //TODO implement SSE and then completed.
    public void train(Matrix features) throws Exception
    {
        features = new Matrix(features,0,1,features.rows(),features.cols()-1);
        //Initialize memory
        intitializeArrays(features.rows());

        //Initialize First Centroids
        initializeCentroids(features);
        printCentroids();

        for(int i=0;i<2;i++)
        {
            //Calculate Distances
            calculateDistances(features);

            //Determine Cluster
            determineCluster();

            //Print Cluster Assignments
            //printAssignments();

            //Calculate New Centroid
            calculateNewCentroid(features);
        }
    }

    private void calculateNewCentroid(Matrix features)
    {
        //Create list if does not exist else clear
        for(int i =0;i<clusters.length;i++)
        {
            if (map.get(clusters[i]) == null)
            {
                map.put(clusters[i], new ArrayList<Integer>());
            }
            else
            {
                map.get(clusters[i]).clear();
            }
        }
        for(int i =0;i<clusters.length;i++)
        {
            map.get(clusters[i]).add(i);
        }
        System.out.println();
        System.out.println("Current Clusters: ");
        printMap();
        System.out.println();
        System.out.println("Calculating new Centroid");

        double SSE =0;
        for (int i = 0; i < features.rows(); i++)
        {
            for (int j = 0; j < features.cols(); j++)
            {
                if(features.valueCount(j) == 0)
                {
                    if(features.get(i,j) == Matrix.MISSING || centroids.get(clusters[i],j) == Matrix.MISSING)
                    {
                        SSE += 1;
                    }
                    else
                    {
                        SSE += Math.pow((features.get(i, j) - centroids.get(clusters[i], j)), 2);
                    }
                }
                else
                {
                    if(features.get(i,j) == Matrix.MISSING || centroids.get(clusters[i],j) == Matrix.MISSING)
                    {
                        SSE += 1;
                    }
                    else
                    {
                        if(features.get(i,j) == centroids.get(clusters[i],j))
                        {
                            SSE +=0;
                        }
                        else
                        {
                            SSE +=1;
                        }
                    }
                }

            }
        }

        //Calculate New Centroid
        //Iterate through each centroid
        for(int i =0;i<map.size();i++)
        {
            //Iterate through all the columns
            for(int j=0;j<features.cols();j++)
            {
                double mean =0;
                int size =0;
                boolean nominal =false;
                //Iterate through each item in centroids
                for (int k = 0; k < map.get(i).size(); k++)
                {
                    if(features.valueCount(j) == 0)
                    {
                        if(features.get((map.get(i).get(k)), j)!= Matrix.MISSING)
                        {
                            mean += features.get((map.get(i).get(k)), j);
                            size ++;
                        }
                    }
                    else if(features.valueCount((j)) !=0)
                    {
                        nominal = true;
                        int [] mostCommonValue = new int[features.valueCount(j)];
                        for(int s =0; s<mostCommonValue.length;s++)
                        {
                            if(features.get((map.get(i).get(k)),j) != Matrix.MISSING)
                            {
                                mostCommonValue[(int) features.get((map.get(i).get(k)), j)]++;
                            }
                        }
                        int largest =0;
                        for(int s =0; s<mostCommonValue.length;s++)
                        {
                            if(mostCommonValue[s]>largest)
                            {
                                largest = mostCommonValue[s];
                                mean = s;
                            }
                        }
                    }
                }
                if(!nominal)
                {
                    mean = mean / size;
                }
                if(Double.isNaN(mean))
                {
                    centroids.set(i,j,Matrix.MISSING);
                }
                else
                {
                    centroids.set(i, j, mean);
                }

            }
        }
        printCentroids();
        System.out.println("SSE: " + SSE);
    }

    private void printMap()
    {
        for(int i =0;i<map.size();i++)
        {
            System.out.println("Cluster " + i + ": " + map.get(i));
        }
    }

    private void printAssignments()
    {
        System.out.println("Making Assignments");
        for(int i=0;i<clusters.length;i++)
        {
//            if(i%10==0)
//            {
//                System.out.println();
//            }
//            System.out.print(i + "=" + clusters[i] + " ");
            System.out.print(clusters[i]+",");
        }
    }

    private void determineCluster()
    {
        for(int i=0;i<distances.length;i++)
        {
            double closestCluster = Double.MAX_VALUE;
            int index = -1;
            for(int j=0;j<distances[0].length;j++)
            {
                if(distances[i][j] < closestCluster)
                {
                    closestCluster = distances[i][j];
                    index = j;
                }
            }
            clusters[i]=index;
        }
    }

    private void calculateDistances(Matrix features)
    {
        for(int i =0; i<features.rows();i++)
        {

            for(int j=0;j<kMeans;j++)
            {
                double distance  = 0;
                for(int k =0;k<features.cols();k++)
                {
                    if(features.get(i,k) == Matrix.MISSING || centroids.get(j,k) == Matrix.MISSING)
                    {
                        distance += 1;
                    }
                    else if(features.valueCount(k)==0)
                    {
                        distance += Math.pow((features.get(i,k) - centroids.get(j,k)),2);
                    }
                    else
                    {
                        if(centroids.get(j,k) - features.get(i,k) !=0)
                        {
                            distance += 1;
                        }
                        else
                        {
                            distance +=0;
                        }
                    }

                }
                distance = Math.sqrt(distance);
                distances[i][j] = distance;
            }
        }
    }

    private void printCentroids()
    {
        for(int i=0; i<centroids.rows();i++)
        {
            String data = "";
            for(int j=0;j<centroids.cols();j++)
            {
                if(j==centroids.cols()-1)
                {
                    if(centroids.get(i,j) == Matrix.MISSING)
                    {
                        data += "?";
                    }
                    else
                    {
                        data += centroids.get(i, j);
                    }
                }
                else
                {
                    if(centroids.get(i,j) == Matrix.MISSING)
                    {
                        data += "? , ";
                    }
                    else
                    {
                        data += centroids.get(i, j) + ", ";
                    }
                }
            }
            System.out.println("Centroid " + i + ": " + data);
        }
    }

    private void initializeCentroids(Matrix features)
    {
        centroids = new Matrix(features,0,0,kMeans,features.cols());
    }

    private void intitializeArrays(int size)
    {
        distances = new double[size][kMeans];
        clusters = new int [size];
    }
}
