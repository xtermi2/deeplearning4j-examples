package org.deeplearning4j.examples.advanced.modelling.textclassification.pretrainedword2vec;

import com.google.common.collect.ImmutableSet;
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream;
import org.nd4j.shade.jackson.databind.JsonNode;
import org.nd4j.shade.jackson.databind.MappingIterator;
import org.nd4j.shade.jackson.databind.json.JsonMapper;
import org.nd4j.shade.jackson.databind.node.ArrayNode;

import javax.xml.stream.XMLInputFactory;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.XMLStreamReader;
import javax.xml.stream.events.XMLEvent;
import java.io.*;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Set;

import static org.apache.commons.lang3.StringUtils.isNotBlank;

public class GenerateName2LableCsvData {

    protected static final String INDIVIDUAL = "Individual";
    protected static final String ORGANIZATION = "Organization";
    private static final Set<String> CLASSIFIERS = ImmutableSet.of(INDIVIDUAL, ORGANIZATION);
    protected static final String OUT_DELIMITER = "|";

    public static void main(String[] args) throws Exception {
        final String outBaseDir = "/home/akeefer/dev/priv/deeplearning4j-examples/dl4j-examples/src/main/resources/";

        final String worldCompliangeCsvOutCsv = outBaseDir + "20190920134848_Entities.csv";
        createWorldCompliangeCsv("/home/akeefer/dev/proxora/worldcompliance_small/20190920134848/20190920134848_Entities.txt",
            worldCompliangeCsvOutCsv);

        final String worldComlinaceXmlOutCsv = outBaseDir + "worldcompliance_proxora_entities.xml.csv";
        createWorldComlinaceXml("/home/akeefer/dev/proxora/worldcompliance_proxora/Entities.xml",
            worldComlinaceXmlOutCsv);

        final String worldCheckXmlOutCsv = outBaseDir + "worldcheck_BPC_F_181119121212.csv";
//        createWorldCheckXml("/tmp/BPC_F_181119121212.xml",
//            worldCheckXmlOutCsv);

        // handelsregister https://offeneregister.de/
        final String handelsregisterDeOutCsv = outBaseDir + "handelsregister_de.csv";
        createHandelsregisterDe(outBaseDir + "de_companies_ocdata.jsonl.bz2",
            handelsregisterDeOutCsv);

        mergeAllOutFilesTo(outBaseDir + "name2type.csv",
            worldCompliangeCsvOutCsv,
            worldComlinaceXmlOutCsv,
            worldCheckXmlOutCsv,
            handelsregisterDeOutCsv);
    }

    private static void mergeAllOutFilesTo(String outMergeFileName,
                                           String... filesToMerge) throws IOException {
        System.out.println();
        System.out.println("merge following files in file " + outMergeFileName + ":");
        System.out.println(Arrays.toString(filesToMerge));
        System.out.println();

        try (final FileWriter fileWriter = new FileWriter(outMergeFileName);
             final PrintWriter printWriter = new PrintWriter(fileWriter)) {
            for (String fileToMerge : filesToMerge) {
                Files.lines(Paths.get(new File(fileToMerge).toURI()))
                    .forEach(printWriter::println);
            }

        }
    }

    private static void createHandelsregisterDe(String inJsonlBz2, String outCsv) throws IOException {
        System.out.println("read " + inJsonlBz2);
        System.out.println();
        final JsonMapper mapper = JsonMapper.builder().build();
        try (final FileWriter fileWriter = new FileWriter(outCsv);
             final PrintWriter printWriter = new PrintWriter(fileWriter);
             final BZip2CompressorInputStream bzIn = new BZip2CompressorInputStream(new FileInputStream(inJsonlBz2), true);
             MappingIterator<JsonNode> it = mapper.readerFor(JsonNode.class).readValues(bzIn)) {

            while (it.hasNextValue()) {
                JsonNode node = it.nextValue();
                {
                    String companyName = node.path("name").asText();
                    printWriter.println(companyName + OUT_DELIMITER + ORGANIZATION);
                }
                final ArrayNode officers = (ArrayNode) node.get("officers");
                if (null != officers) {
                    for (JsonNode officer : officers) {
                        final String name = officer.get("name").asText();
                        String type = officer.get("type").asText();
                        if ("person".equals(type)) {
                            type = INDIVIDUAL;
                        } else if ("company".equals(type)) {
                            type = ORGANIZATION;
                        } else {
                            System.err.println("unknown type: " + type);
                            type = null;
                        }
                        if (null != type) {
                            printWriter.println(name + OUT_DELIMITER + type);
                        }
                    }
                }
            }
        }
    }

    private static void createWorldCompliangeCsv(String inCsv, String outCsv) throws URISyntaxException, IOException {
        System.out.println("read " + inCsv);
        System.out.println();
        final int nameIndex = 1;
        final int typeIndex = 10;
        final String inDelimiter = "|";

        try (final FileWriter fileWriter = new FileWriter(outCsv);
             final PrintWriter printWriter = new PrintWriter(fileWriter)) {

            Files.lines(Paths.get(new File(inCsv).toURI()))
                .map(line -> line.split("\\" + inDelimiter))
                .filter(split -> split.length > typeIndex && CLASSIFIERS.contains(split[typeIndex]))
                .map(split -> split[nameIndex] + OUT_DELIMITER + split[typeIndex])
                .distinct()
                .forEach(printWriter::println);
        }
    }

    private static void createWorldCheckXml(String inXml, String outCsv) throws IOException, XMLStreamException {
        System.out.println("read " + inXml);
        System.out.println();
        XMLStreamReader reader = XMLInputFactory.newInstance()
            .createXMLStreamReader(new FileInputStream(inXml));

        try (FileWriter fileWriter = new FileWriter(outCsv);
             PrintWriter printWriter = new PrintWriter(fileWriter)) {


            String lastName = null;
            String firstName = null;
            String type = null;
            boolean inEntity = false;
            while (reader.hasNext()) {
                int event = reader.next();

                if (event == XMLEvent.START_ELEMENT
                    && "person".equals(reader.getLocalName())) {
                    inEntity = true;
                    for (int i = 0; i < reader.getAttributeCount(); i++) {
                        if ("e-i".equals(reader.getAttributeLocalName(i))) {
                            type = reader.getAttributeValue(i);
                            if ("E".equals(type)) {
                                type = ORGANIZATION;
                            } else if ("I".equals(type)
                                || "F".equals(type)
                                || "U".equals(type)
                                || "M".equals(type)) {
                                type = INDIVIDUAL;
                            } else {
                                type = null;
                            }
                        }
                    }
                } else if (event == XMLEvent.END_ELEMENT
                    && "person".equals(reader.getLocalName())) {
                    inEntity = false;
                    if (isNotBlank(lastName) && isNotBlank(type)) {
                        String name = isNotBlank(firstName)
                            ? lastName + ", " + firstName
                            : lastName;
                        printWriter.println(name + OUT_DELIMITER + type);
                    }
                    firstName = null;
                    lastName = null;
                    type = null;
                } else if (inEntity &&
                    event == XMLEvent.START_ELEMENT
                    && "first_name".equals(reader.getLocalName())) {
                    firstName = reader.getElementText();
                } else if (inEntity &&
                    event == XMLEvent.START_ELEMENT
                    && "last_name".equals(reader.getLocalName())) {
                    lastName = reader.getElementText();
                }
            }
        }
    }

    private static void createWorldComlinaceXml(String inXml, String outCsv) throws IOException, XMLStreamException {
        System.out.println("read " + inXml);
        System.out.println();
        XMLStreamReader reader = XMLInputFactory.newInstance()
            .createXMLStreamReader(new FileInputStream(inXml));

        try (FileWriter fileWriter = new FileWriter(outCsv);
             PrintWriter printWriter = new PrintWriter(fileWriter)) {


            String name = null;
            String type = null;
            boolean inEntity = false;
            while (reader.hasNext()) {
                int event = reader.next();

                if (event == XMLEvent.START_ELEMENT
                    && "Entity".equals(reader.getLocalName())) {
                    inEntity = true;
                } else if (event == XMLEvent.END_ELEMENT
                    && "Entity".equals(reader.getLocalName())) {
                    inEntity = false;
                    if (isNotBlank(name) && isNotBlank(type)) {
                        printWriter.println(name + OUT_DELIMITER + type);
                    }
                    name = null;
                    type = null;
                } else if (event == XMLEvent.END_ELEMENT
                    && "Entities".equals(reader.getLocalName())) {
                    break;
                } else if (inEntity &&
                    event == XMLEvent.START_ELEMENT
                    && "Name".equals(reader.getLocalName())) {
                    name = reader.getElementText();
                } else if (inEntity &&
                    event == XMLEvent.START_ELEMENT
                    && "EntityTypeDesc".equals(reader.getLocalName())) {
                    final String elementText = reader.getElementText();
                    type = CLASSIFIERS.contains(elementText)
                        ? elementText
                        : null;
                }
            }
        }
    }
}
