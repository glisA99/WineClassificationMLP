package rs.ac.bg.fon.is.winequalityclassification;

import java.io.File;
import java.util.Arrays;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.learning.error.MeanSquaredError;
import org.neuroph.eval.ClassifierEvaluator;
import org.neuroph.eval.ErrorEvaluator;
import org.neuroph.eval.Evaluation;
import org.neuroph.eval.classification.ClassificationMetrics;
import org.neuroph.eval.classification.ConfusionMatrix;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.TransferFunctionType;
import org.neuroph.util.data.norm.MaxNormalizer;
import org.neuroph.util.data.norm.Normalizer;

/**
 * Data set that will be used in this experiment: Wine Quality Dataset The Wine
 * Quality Dataset involves predicting the quality of white wines on a scale
 * given chemical measures of each wine. It is a multi-class classification
 * problem, but could also be framed as a regression problem. The original data
 * set that will be used in this experiment can be found at link:
 * http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv
 *
 * Number of instances: 4 898  *
 * Number of Attributes: 11 plus class attributes (inputs are continuous and
 * numerical values, and output is numerical)
 *
 * Attribute Information: Inputs: 11 attributes: 11 numerical or continuous
 * features are computed for each wine: 1) Fixed acidity. 2) Volatile acidity.
 * 3) Citric acid. 4) Residual sugar. 5) Chlorides. 6) Free sulfur dioxide. 7)
 * Total sulfur dioxide. 8) Density. 9) pH. 10) Sulphates. 11) Alcohol.
 *
 * 12) Output: Quality (score between 0 and 10)
 *
 * Missing Values: None.
 *
 */
/**
 *
 * @author Ognjen Simic
 */
public class Classification {

    public static void main(String[] args) {
        //System.out.println((new File("rand")).getAbsolutePath());
        String dataset_path = "./dataset/wine_classification_data.txt";
        int input_count = 11;
        int output_count = 3;

        // create dataset
        System.out.println("Creating dataset...");
        DataSet dataset = DataSet.createFromFile(dataset_path, input_count, output_count, "\t",false);

        // Normalize dataset using max normalization: normalizedVector[i] = vector[i] / abs(max[i])
        Normalizer normalizer = new MaxNormalizer(dataset);
        normalizer.normalize(dataset);

        // Shufle dataset
        dataset.shuffle();

        // split data into train and test set
        DataSet[] split = dataset.split(0.6, 0.4);
        DataSet train_data = split[0];
        DataSet test_data = split[1];

        System.out.println("Creating neural networks...");
        int HIDDEN_COUNT_1 = 20;
        int HIDDEN_COUNT_2 = 15;
        double MAX_ERROR = 0.02;
        double[] learning_rates = new double[]{0.2, 0.4, 0.6};
        MultiLayerPerceptron[] nnets = new MultiLayerPerceptron[3];
        for (int i = 0; i < nnets.length; i++) {
            nnets[i] = new MultiLayerPerceptron(
                    input_count,
                    HIDDEN_COUNT_1,
                    HIDDEN_COUNT_2,
                    output_count);
            nnets[i].setLearningRule(new MomentumBackpropagation());
            MomentumBackpropagation lrule = (MomentumBackpropagation) nnets[i].getLearningRule();
            lrule.addListener((event) -> {
                MomentumBackpropagation mbp = (MomentumBackpropagation) event.getSource();
                System.out.println(mbp.getCurrentIteration() + ". iteration | Total network error: "
                        + mbp.getTotalNetworkError());
            });
            lrule.setMaxError(MAX_ERROR);
            lrule.setLearningRate(learning_rates[i]);
        }
        
        int total_iterations = 0;
        for(int i=0;i<nnets.length;i++) {
            System.out.println("Training " + i + ". nnet:");
            nnets[i].learn(train_data);
            MomentumBackpropagation lrule = (MomentumBackpropagation) nnets[i].getLearningRule();
            System.out.println("\nNNET WITH LEARNING_RATE=" + learning_rates[i] + " | " + 
                    "Number of iterations: " + lrule.getCurrentIteration() + "\n");
            total_iterations += lrule.getCurrentIteration();
        }
        System.out.println("Mean number of iterations for all nnets: " + total_iterations/nnets.length);
        
        double bestAccuracy = 0;
        int index = 0;
        for(int i=0;i<nnets.length;i++) {
            testNeuralNetwork(test_data,nnets[i]);
            ClassificationMetrics.Stats stat = evaluate(test_data, nnets[i]);
            if (stat.accuracy > bestAccuracy) {
                bestAccuracy = stat.accuracy;
                index = i;
            }
        }
        System.out.println("NNET with best accuracy was with learning rate=" + learning_rates[index]);
        
        // save best nnet
        nnets[index].save("best_nnet");
    }
    
    public static void testNeuralNetwork(DataSet testSet, NeuralNetwork nnet) {
        for(DataSetRow row : testSet.getRows()) {
            nnet.setInput(row.getInput());
            nnet.calculate();
            double[] output = nnet.getOutput();
            
            System.out.println("Input: " + Arrays.toString(row.getInput()));
            System.out.println("Output: " + Arrays.toString(output));
            System.out.println("Desired output" + Arrays.toString(row.getDesiredOutput()));
        }
    }
    
    // Evaluates performance of neural network
    public static ClassificationMetrics.Stats evaluate(DataSet testSet, NeuralNetwork nnet) {
        System.out.println("Calculating performance indicators for neural network.");
        Evaluation evaluation = new Evaluation();
        evaluation.addEvaluator(new ErrorEvaluator(new MeanSquaredError()));
        
        String classLabels[] = new String[]{"1", "2", "3", "4", "5", "6", "7", "8", "9", "10"};
        evaluation.addEvaluator(new ClassifierEvaluator.MultiClass(classLabels));
        evaluation.evaluate(nnet, testSet);
        
        ClassifierEvaluator.MultiClass evaluator = evaluation.getEvaluator(ClassifierEvaluator.MultiClass.class);
        ConfusionMatrix confusionMatrix = evaluator.getResult();
        System.out.println(confusionMatrix);
        ClassificationMetrics[] metrics = ClassificationMetrics.createFromMatrix(confusionMatrix);
        ClassificationMetrics.Stats average = ClassificationMetrics.average(metrics);
        for(ClassificationMetrics metric : metrics) {
            System.out.println(metric + "\n");
        }
        System.out.println("AVERAGE: " + average);
        return average;
    }

}
