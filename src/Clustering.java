import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

class Clustering
{
    private int kMeans = 5;
    private double[][] distances;
    private int[] clusters;
    private Matrix centroids;
    private HashMap<Integer, List<Integer>> map = new HashMap<Integer, List<Integer>>();
    boolean run = true;
    double SSE = 0;
    double previousSSE = 0;
    int runCount = 0;

    DecimalFormat df = new DecimalFormat("#.###");


    //TODO implement SSE and then completed.
    public void train(Matrix features) throws Exception
    {
        df.setRoundingMode(RoundingMode.CEILING); //Used for rounding in strings for printing values
        //features = new Matrix(features, 0, 1, features.rows(), features.cols() - 1); //Only used for labor data comment out if not testing on labor data.
        //Initialize memory
        intitializeArrays(features.rows());

        //Initialize First Centroids
        initializeCentroids(features);

        //Run program until SSE convergence is met, meaning clusters are no longer moving.
        while (run)
        {
            //Calculate Distances
            calculateDistances(features);

            //Determine Cluster
            determineCluster();

            //Calculate New Centroid
            calculateNewCentroid(features);

            if(runCount > 2)
            {
                previousSSE = Math.round(previousSSE);
                SSE = Math.round(SSE);
                if(previousSSE == SSE)
                {
                    run = false;
                }


            }
            previousSSE = SSE;
            runCount++;
        }
        System.out.println("SSE has converged stopping program");
    }

    //Bulk of the program calculate the new centroid of cluster.
    private void calculateNewCentroid(Matrix features)
    {
        //Create list if does not exist else clear
        for (int i = 0; i < clusters.length; i++)
        {
            if (map.get(clusters[i]) == null)
            {
                map.put(clusters[i], new ArrayList<Integer>());
            } else
            {
                map.get(clusters[i]).clear();
            }
        }

        //Add the feature row to the clusters
        for (int i = 0; i < clusters.length; i++)
        {
            map.get(clusters[i]).add(i);
        }
        printCentroids();
        System.out.println();
        System.out.println("Current Clusters: ");
        printMap();

        //Calculate SSE
        SSE = 0;
        for (int i = 0; i < features.rows(); i++)
        {
            for (int j = 0; j < features.cols(); j++)
            {
                //If continuous Data
                if (features.valueCount(j) == 0)
                {
                    //If missing SSE +1
                    if (features.get(i, j) == Matrix.MISSING || centroids.get(clusters[i], j) == Matrix.MISSING)
                    {
                        SSE += 1;
                    } else
                    {
                        SSE += Math.pow((features.get(i, j) - centroids.get(clusters[i], j)), 2);
                    }
                }
                //Nominal Data
                else
                {
                    if (features.get(i, j) == Matrix.MISSING || centroids.get(clusters[i], j) == Matrix.MISSING)
                    {
                        SSE += 1;
                    } else
                    {
                        if (features.get(i, j) == centroids.get(clusters[i], j))
                        {
                            SSE += 0;
                        } else
                        {
                            SSE += 1;
                        }
                    }
                }

            }
        }
        System.out.println("SSE: " + SSE);
        System.out.println();
        System.out.println("Calculating new Centroid");
        //Calculate New Centroid
        //Iterate through each centroid
        for (int i = 0; i < map.size(); i++)
        {
            //Iterate through all the columns
            for (int j = 0; j < features.cols(); j++)
            {
                int[] mostCommonValue = new int[features.valueCount(j)];
                int largest = -1;
                double mean = 0;
                int size = 0;
                boolean nominal = false;
                //Iterate through each item in centroids
                for (int k = 0; k < map.get(i).size(); k++)
                {
                    int row = map.get(i).get(k);
                    int value = j;
                    if (features.valueCount(j) == 0)
                    {
                        if (features.get((map.get(i).get(k)), j) != Matrix.MISSING)
                        {
                            mean += features.get((map.get(i).get(k)), j);
                            size++;
                        }
                    } else if (features.valueCount((j)) != 0)
                    {
                        nominal = true;

                        if (features.get((map.get(i).get(k)), j) != Matrix.MISSING)
                        {
                            mostCommonValue[(int) features.get((map.get(i).get(k)), j)]++;
                        }

                    }
                    for (int s = 0; s < mostCommonValue.length; s++)
                    {
                        if (mostCommonValue[s] > largest)
                        {
                            largest = mostCommonValue[s];
                            mean = s;
                        }
                    }
                }
                if (!nominal)
                {
                    mean = mean / size;
                }
                if (Double.isNaN(mean))
                {
                    centroids.set(i, j, Matrix.MISSING);
                } else
                {
                    centroids.set(i, j, mean);
                }
            }
        }

    }

    //Function for printing the clusters
    private void printMap()
    {
        for (int i = 0; i < map.size(); i++)
        {
            System.out.println("Cluster " + i + ": " + map.get(i));
        }
    }

    //Assign clusters based on cluster that is closest
    private void determineCluster()
    {
        for (int i = 0; i < distances.length; i++)
        {
            double closestCluster = Double.MAX_VALUE;
            int index = -1;
            for (int j = 0; j < distances[0].length; j++)
            {
                if (distances[i][j] < closestCluster)
                {
                    closestCluster = distances[i][j];
                    index = j;
                }
            }
            clusters[i] = index;
        }
    }

    //Distance function used Euclidian distance
    private void calculateDistances(Matrix features)
    {
        for (int i = 0; i < features.rows(); i++)
        {

            for (int j = 0; j < kMeans; j++)
            {
                double distance = 0;
                for (int k = 0; k < features.cols(); k++)
                {
                    if (features.get(i, k) == Matrix.MISSING || centroids.get(j, k) == Matrix.MISSING)
                    {
                        distance += 1;
                    } else if (features.valueCount(k) == 0)
                    {
                        distance += Math.pow((features.get(i, k) - centroids.get(j, k)), 2);
                    } else
                    {
                        if (centroids.get(j, k) != features.get(i, k))
                        {
                            distance += 1;
                        } else
                        {
                            distance += 0;
                        }
                    }

                }
                distance = Math.sqrt(distance);
                distances[i][j] = distance;
            }
        }
    }

    //Function to print the centroids. This does not tie back into the Matrix feature to show the names for nominal data
    private void printCentroids()
    {
        for (int i = 0; i < centroids.rows(); i++)
        {
            String data = "";
            for (int j = 0; j < centroids.cols(); j++)
            {
                if (j == centroids.cols() - 1)
                {
                    if (centroids.get(i, j) == Matrix.MISSING)
                    {
                        data += "?";
                    } else
                    {
                        data += df.format(centroids.get(i, j));
                    }
                } else
                {
                    if (centroids.get(i, j) == Matrix.MISSING)
                    {
                        data += "? , ";
                    } else
                    {
                        data += df.format(centroids.get(i, j)) + ", ";
                    }
                }
            }
            System.out.println("Centroid " + i + ": " + data);
        }
    }

    //Memory allocation functions to set intial centroids and initialize arrays.
    private void initializeCentroids(Matrix features)
    {
        centroids = new Matrix(features, 0, 0, kMeans, features.cols());
    }

    private void intitializeArrays(int size)
    {
        distances = new double[size][kMeans];
        clusters = new int[size];
    }
}
