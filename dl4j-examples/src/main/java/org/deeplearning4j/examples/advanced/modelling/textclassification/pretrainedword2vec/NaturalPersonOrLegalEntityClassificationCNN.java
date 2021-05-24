/* *****************************************************************************
 * Copyright (c) 2020 Konduit K.K.
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.examples.advanced.modelling.textclassification.pretrainedword2vec;

import com.clearspring.analytics.util.Pair;
import com.diogonunes.jcolor.Ansi;
import com.google.common.base.Stopwatch;
import com.google.common.collect.ImmutableSet;
import org.deeplearning4j.iterator.CnnSentenceDataSetIterator;
import org.deeplearning4j.iterator.CnnSentenceDataSetIterator.Format;
import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.deeplearning4j.iterator.provider.CollectionLabeledSentenceProvider;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.NGramTokenizerFactory;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static com.diogonunes.jcolor.Attribute.GREEN_TEXT;
import static com.diogonunes.jcolor.Attribute.RED_TEXT;

/**
 * Convolutional Neural Networks for Sentence Classification - https://arxiv.org/abs/1408.5882
 * <p>
 * Specifically, this is the 'static' model from there
 *
 * @author Andreas Keefer
 */
public class NaturalPersonOrLegalEntityClassificationCNN {

    private static final boolean train = false;
    private static final boolean saveAfterTrain = false;
    private static final String networkSaveName = "/home/akeefer/dev/priv/deeplearning4j-examples/dl4j-examples/src/main/resources/NaturalPersonOrLegalEntityClassificationCNN.zip";

    private static final Set<String> CLASSIFIERS = ImmutableSet.of("Individual", "Organization");

    public static void main(String[] args) throws Exception {
        final Pair<Pair<List<String>, List<String>>, Pair<List<String>, List<String>>> trainingAndTestSentence2Label =
            loadTrainingAndTestSentence2Label(
                "name2type.csv", "\\|", 3);

        Stopwatch stopwatchPrepareWords2Vector = Stopwatch.createStarted();
        if (ImdbReviewClassificationRNN.wordVectorsPath.startsWith("/PATH/TO/YOUR/VECTORS/")) {
            System.out.println("wordVectorsPath has not been set. Checking default location in ~/dl4j-examples-data for download...");
            ImdbReviewClassificationRNN.checkDownloadW2VECModel();
        }

        //Download and extract data
        ImdbReviewClassificationRNN.downloadData();

        //Basic configuration
        int batchSize = 32;
        int vectorSize = 300;               //Size of the word vectors. 300 in the Google News model
        int nEpochs = 1;                    //Number of epochs (full passes of training data) to train on
        int truncateNameToLength = 64;  //Truncate Name with length (# words) greater than this
        Random rng = new Random(12345); //For shuffling repeatability

        //Load word vectors and get the DataSetIterators for training and testing
        System.out.println("Loading word vectors and creating DataSetIterators...");
        WordVectors wordVectors = WordVectorSerializer.loadStaticModel(new File(ImdbReviewClassificationRNN.wordVectorsPath));

        DataSetIterator testIter = getDataSetIterator(trainingAndTestSentence2Label.right, wordVectors, batchSize, truncateNameToLength, rng);
        stopwatchPrepareWords2Vector.stop();
        System.out.println("Loading word vectors and creating DataSetIterators took " + stopwatchPrepareWords2Vector);

        final ComputationGraph net;
        if (train) {
            final Stopwatch stopwatchTraining = Stopwatch.createStarted();
            int cnnLayerFeatureMaps = 1000;      //Number of feature maps / channels / depth for each CNN layer
            PoolingType globalPoolingType = PoolingType.MAX;

            //Set up the network configuration. Note that we have multiple convolution layers, each wih filter
            //widths of 3, 4 and 5 as per Kim (2014) paper.

            Nd4j.getMemoryManager().setAutoGcWindow(5000);

            ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.RELU)
                .activation(Activation.LEAKYRELU)
                .updater(new Adam(0.01))
                .convolutionMode(ConvolutionMode.Same)      //This is important so we can 'stack' the results later
                .l2(0.0001)
                .graphBuilder()
                .addInputs("input")
                .addLayer("cnn3", new ConvolutionLayer.Builder()
                    .kernelSize(3, vectorSize)
                    .stride(1, vectorSize)
                    .nOut(cnnLayerFeatureMaps)
                    .build(), "input")
                .addLayer("cnn4", new ConvolutionLayer.Builder()
                    .kernelSize(4, vectorSize)
                    .stride(1, vectorSize)
                    .nOut(cnnLayerFeatureMaps)
                    .build(), "input")
                .addLayer("cnn5", new ConvolutionLayer.Builder()
                    .kernelSize(5, vectorSize)
                    .stride(1, vectorSize)
                    .nOut(cnnLayerFeatureMaps)
                    .build(), "input")
                //MergeVertex performs depth concatenation on activations: 3x[minibatch,100,length,300] to 1x[minibatch,300,length,300]
                .addVertex("merge", new MergeVertex(), "cnn3", "cnn4", "cnn5")
                //Global pooling: pool over x/y locations (dimensions 2 and 3): Activations [minibatch,300,length,300] to [minibatch, 300]
                .addLayer("globalPool", new GlobalPoolingLayer.Builder()
                    .poolingType(globalPoolingType)
                    .dropOut(0.5)
                    .build(), "merge")
                .addLayer("out", new OutputLayer.Builder()
                    .lossFunction(LossFunctions.LossFunction.MCXENT)
                    .activation(Activation.SOFTMAX)
                    .nOut(2)    //2 classes: positive or negative
                    .build(), "globalPool")
                .setOutputs("out")
                //Input has shape [minibatch, channels=1, length=1 to 256, 300]
                .setInputTypes(InputType.convolutional(truncateNameToLength, vectorSize, 1))
                .build();

            net = new ComputationGraph(config);
            net.init();

            System.out.println("Number of parameters by layer:");
            for (Layer l : net.getLayers()) {
                System.out.println("\t" + l.conf().getLayer().getLayerName() + "\t" + l.numParams());
            }

            DataSetIterator trainIter = getDataSetIterator(trainingAndTestSentence2Label.left, wordVectors, batchSize, truncateNameToLength, rng);

            System.out.println("Starting training");
            net.setListeners(new ScoreIterationListener(100), new EvaluativeListener(testIter, 1, InvocationType.EPOCH_END));
            net.fit(trainIter, nEpochs);
            if (saveAfterTrain) {
                System.out.println("save Network to file " + networkSaveName);
                net.save(new File(networkSaveName), true);
            }
            stopwatchTraining.stop();
            System.out.println("Training took " + stopwatchTraining);
        } else {
            System.out.println("load Network from file " + networkSaveName);
            net = ComputationGraph.load(new File(networkSaveName), true);
        }

        for (int i = 0; i < 50; i++) {
            String sentence = trainingAndTestSentence2Label.right.left.get(i);
            String label = trainingAndTestSentence2Label.right.right.get(i);
            System.out.println("\nPrediction for " + label + " '" + sentence + "':");
            final Optional<Pair<Double, String>> prediction = makePrediction(net, testIter, sentence);
            prediction.ifPresent(p -> {
                if (Objects.equals(label, p.right)) {
                    System.out.println(Ansi.colorize(" [Correct Prediction]", GREEN_TEXT()));
                } else {
                    System.out.println(Ansi.colorize(" [Wrong Prediction]", RED_TEXT()));
                }
            });
        }
        for (int i = trainingAndTestSentence2Label.right.left.size() - 1; i > trainingAndTestSentence2Label.right.left.size() - 50; i--) {
            String sentence = trainingAndTestSentence2Label.right.left.get(i);
            String label = trainingAndTestSentence2Label.right.right.get(i);
            System.out.println("\nPrediction for " + label + " '" + sentence + "':");
            final Optional<Pair<Double, String>> prediction = makePrediction(net, testIter, sentence);
            prediction.ifPresent(p -> {
                if (Objects.equals(label, p.right)) {
                    System.out.println(Ansi.colorize(" [Correct Prediction]", GREEN_TEXT()));
                } else {
                    System.out.println(Ansi.colorize(" [Wrong Prediction]", RED_TEXT()));
                }
            });
        }

        Thread.sleep(500);
        try (BufferedReader br = new BufferedReader(new InputStreamReader(System.in))) {
            String name = "";
            while (!"q".equals(name)) {
                System.out.println("\n\nEnter a name (Individual or Organization or 'q' to exit):");
                name = br.readLine();
                if (!"q".equals(name)) {
                    makePrediction(net, testIter, name);
                }
            }
        }
    }

    private static Optional<Pair<Double, String>> makePrediction(ComputationGraph net, DataSetIterator testIter, String sentence) {
        try {
            final Stopwatch stopwatch = Stopwatch.createStarted();
            INDArray features = ((CnnSentenceDataSetIterator) testIter).loadSingleSentence(sentence);
            INDArray predictions = net.outputSingle(features);
            List<String> labels = testIter.getLabels();
            stopwatch.stop();

            Pair<Double, String> highestPrediction = null;
            for (int j = 0; j < labels.size(); j++) {
                String label = labels.get(j);
                final double value = predictions.getDouble(j);
                if (null == highestPrediction || highestPrediction.left < value) {
                    highestPrediction = new Pair<>(value, label);
                }
            }
            BigDecimal percent = BigDecimal.valueOf(highestPrediction.left * 100d).setScale(5, RoundingMode.HALF_UP);
            System.out.print(percent + "% " + highestPrediction.right + " (took " + stopwatch.elapsed(TimeUnit.MILLISECONDS) + "ms)");
            return Optional.of(highestPrediction);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return Optional.empty();
    }

    private static Pair<Pair<List<String>, List<String>>, Pair<List<String>, List<String>>> loadTrainingAndTestSentence2Label(
        final String filename,
        final String delimiter,
        final int moduloTrainigToTest) throws URISyntaxException, IOException {

        Stopwatch stopwatch = Stopwatch.createStarted();
        System.out.println("loading data from '" + filename + "' using '" + delimiter + "' as delimiter regexp");
        Pair<List<String>, List<String>> training = new Pair<>(new ArrayList<>(2500000), new ArrayList<>(2500000));
        Pair<List<String>, List<String>> test = new Pair<>(new ArrayList<>(1220000), new ArrayList<>(1220000));

        AtomicInteger i = new AtomicInteger(0);
        Files.lines(Paths.get(ClassLoader.getSystemResource(filename).toURI()))
            .distinct()
            .map(line -> line.split(delimiter))
            .filter(split -> split.length == 2 && CLASSIFIERS.contains(split[1]))
            .sorted(Comparator.comparing(split -> split[1]))
            .forEach(split -> {
                if (i.getAndIncrement() % moduloTrainigToTest == 0) {
                    //System.out.println("test: " + Arrays.toString(split));
                    test.left.add(split[0]);
                    test.right.add(split[1]);
                } else {
                    //System.out.println("train: " + Arrays.toString(split));
                    training.left.add(split[0]);
                    training.right.add(split[1]);
                }
            });

        stopwatch.stop();
        System.out.println("loadTrainingAndTestSentence2Label took " + stopwatch);
        System.out.println("Count Training data: " + training.left.size() + "/" + training.right.size());
        System.out.println("Count Test data: " + test.left.size() + "/" + test.right.size() + "\n\n");
        return new Pair<>(training, test);
    }


    private static DataSetIterator getDataSetIterator(Pair<List<String>, List<String>> sentence2Label,
                                                      WordVectors wordVectors,
                                                      int minibatchSize,
                                                      int maxSentenceLength,
                                                      Random rng) {


//        String path = FilenameUtils.concat(DATA_PATH, (isTraining ? "train/" : "test/"));
//        String individualFileName = FilenameUtils.concat(path, "individual.txt");
//        String orgFileName = FilenameUtils.concat(path, "org.txt");
//        System.out.println("individualFileName: " + individualFileName);
//        System.out.println("orgFileName: " + individualFileName);

//        File fileIndividual = new File(ClassLoader.getSystemResource(individualFileName).getFile());
//        File fileOrg = new File(ClassLoader.getSystemResource(orgFileName).getFile());

//        Map<String, List<File>> reviewFilesMap = new HashMap<>();
//        reviewFilesMap.put("Individual", Arrays.asList(Objects.requireNonNull(fileIndividual.listFiles())));
//        reviewFilesMap.put("Organisation", Arrays.asList(Objects.requireNonNull(fileOrg.listFiles())));

//        final List<String> individuals = Files.readAllLines(Paths.get(ClassLoader.getSystemResource(individualFileName).toURI()));
//        final List<String> orgs = Files.readAllLines(Paths.get(ClassLoader.getSystemResource(orgFileName).toURI()));
//        final List<String> labels = Stream.concat(
//            IntStream.range(0, individuals.size()).mapToObj(value -> "Individual"),
//            IntStream.range(0, orgs.size()).mapToObj(value -> "Organisation")
//        ).collect(Collectors.toList());
//        List<String> sentences = new ArrayList<>(individuals.size() + orgs.size());
//        sentences.addAll(individuals);
//        sentences.addAll(orgs);

        LabeledSentenceProvider sentenceProvider = new CollectionLabeledSentenceProvider(sentence2Label.left, sentence2Label.right, rng);
//            new FileLabeledSentenceProvider(reviewFilesMap, rng);

        return new CnnSentenceDataSetIterator.Builder(Format.CNN2D)
            .sentenceProvider(sentenceProvider)
            .wordVectors(wordVectors)
            .minibatchSize(minibatchSize)
            .maxSentenceLength(maxSentenceLength)
            .useNormalizedWordVectors(false)
            .unknownWordHandling(CnnSentenceDataSetIterator.UnknownWordHandling.RemoveWord)
            .tokenizerFactory(new NGramTokenizerFactory(new DefaultTokenizerFactory(), 1, 10))
            .build();
    }

}
