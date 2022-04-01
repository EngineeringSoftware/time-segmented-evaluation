package org.tseval;

import com.github.javaparser.ParseProblemException;
import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.stream.JsonWriter;
import org.tseval.MethodDataCollectorVisitor.FilteredReason;
import org.tseval.data.MethodData;
import org.tseval.data.RevisionIds;
import org.tseval.util.BashUtils;
import org.tseval.util.AbstractConfig;
import org.tseval.util.Option;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

public class MethodDataCollector {
    
    public static class Config extends AbstractConfig {
        
        // A list of shas, separated by space
        @Option public String shas;
        @Option public Path projectDir;
        @Option public Path outputDir;
        @Option public Path logFile;
        
        public boolean repOk() {
            if (projectDir == null || outputDir == null || logFile == null) {
                return false;
            }
            
            if (shas == null || shas.length() == 0) {
                return false;
            }
            
            return true;
        }
    }
    public static Config sConfig;
    
    public static final Gson GSON;
    public static final Gson GSON_NO_PPRINT;
    static {
        GsonBuilder gsonBuilder = new GsonBuilder()
                .disableHtmlEscaping()
                .serializeNulls()
                .registerTypeAdapter(MethodData.class, MethodData.sSerDeser)
                .registerTypeAdapter(RevisionIds.class, RevisionIds.sSerializer);
        GSON_NO_PPRINT = gsonBuilder.create();
        gsonBuilder.setPrettyPrinting();
        GSON = gsonBuilder.create();
    }

    private static Map<Integer, Integer> sMethodDataIdHashMap = new HashMap<>();
    private static int sCurrentMethodDataId = 0;
    private static Map<Integer, List<Integer>> sFileCache = new HashMap<>();
    private static Map<FilteredReason, Integer> sFilteredCounters = MethodDataCollectorVisitor.initFilteredCounters();
    
    private static JsonWriter sMethodDataWriter;
    private static JsonWriter sMethodProjectRevisionWriter;
    
    public static void main(String... args) {
        if (args.length != 1) {
            System.err.println("Exactly one argument, the path to the json config, is required");
            System.exit(-1);
        }
        
        sConfig = AbstractConfig.load(Paths.get(args[0]), Config.class);
        collect();
    }
    
    public static void collect() {
        try {
            // Init the writers for saving
            sMethodDataWriter = GSON.newJsonWriter(Files.newBufferedWriter(sConfig.outputDir.resolve("method-data.json")));
            sMethodDataWriter.beginArray();
            sMethodProjectRevisionWriter = GSON_NO_PPRINT.newJsonWriter(Files.newBufferedWriter(sConfig.outputDir.resolve("revision-ids.json")));
            sMethodProjectRevisionWriter.beginArray();
            
            // Collect for each sha (chronological order)
            for (String sha: sConfig.shas.split(" ")) {
                log("Sha " + sha);
                collectSha(sha);
            }
            
            // Save filtered counters
            JsonWriter filteredCountersWriter = GSON.newJsonWriter(Files.newBufferedWriter(sConfig.outputDir.resolve("filtered-counters.json")));
            filteredCountersWriter.beginObject();
            for (FilteredReason fr : FilteredReason.values()) {
                filteredCountersWriter.name(fr.getKey());
                filteredCountersWriter.value(sFilteredCounters.get(fr));
            }
            filteredCountersWriter.endObject();
            filteredCountersWriter.close();

            // Close writers
            sMethodDataWriter.endArray();
            sMethodDataWriter.close();
            sMethodProjectRevisionWriter.endArray();
            sMethodProjectRevisionWriter.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
    
    public static void log(String msg) {
        if (sConfig.logFile != null) {
            try (BufferedWriter fw = Files.newBufferedWriter(sConfig.logFile, StandardOpenOption.APPEND, StandardOpenOption.CREATE)) {
                fw.write("[" + Thread.currentThread().getId() + "]" + msg + "\n");
//                (new Throwable()).printStackTrace(fos);
            } catch (IOException e) {
                System.err.println("Couldn't log to " + sConfig.logFile);
                System.exit(-1);
            }
        }
    }

    private static void collectSha(String sha) throws IOException {
        // Check out the sha
        BashUtils.run("cd " + sConfig.projectDir + " && git checkout -f " + sha, 0);

        // Find all java files
        List<Path> javaFiles = Files.walk(sConfig.projectDir)
                .filter(Files::isRegularFile)
                .filter(p -> p.toString().endsWith(".java"))
                .sorted(Comparator.comparing(Object::toString))
                .collect(Collectors.toList());
        log("In revision " + sha +", got " + javaFiles.size() + " files to parse");

        // For each java file, parse and get methods
        MethodDataCollectorVisitor visitor = new MethodDataCollectorVisitor();
        List<Integer> idsRevision = new LinkedList<>();
        int parseErrorCount = 0;
        int filteredCount = 0;
        int reuseFileCount = 0;
        int parseFileCount = 0;
        for (Path javaFile : javaFiles) {
            // Skip parsing identical files, just add the ids
            int fileHash = getFileHash(javaFile);
            List<Integer> idsFile = sFileCache.get(fileHash);

            if (idsFile == null) {
                // Actually parse this file and collect ids
                idsFile = new LinkedList<>();
                String path = sConfig.projectDir.relativize(javaFile).toString();

                MethodDataCollectorVisitor.Context context = new MethodDataCollectorVisitor.Context();
                try {
                    CompilationUnit cu = StaticJavaParser.parse(javaFile);
                    cu.accept(visitor, context);
                } catch (ParseProblemException e) {
                    ++parseErrorCount;
                    continue;
                }

                for (FilteredReason fr : FilteredReason.values()) {
                    sFilteredCounters.compute(fr, (k, v) -> v + context.filteredCounters.get(k));
                    filteredCount += context.filteredCounters.get(fr);
                }

                for (MethodData methodData : context.methodDataList) {
                    // Reuse (for duplicate data) or allocate the data id
                    methodData.path = path;
                    int methodId = addMethodData(methodData);
                    idsFile.add(methodId);
                }

                // Update file cache
                sFileCache.put(fileHash, idsFile);
                ++parseFileCount;
            } else {
                ++reuseFileCount;
            }

            idsRevision.addAll(idsFile);
        }

        // Create and save MethodProjectRevision
        RevisionIds revisionIds = new RevisionIds();
        revisionIds.revision = sha;
        revisionIds.methodIds = idsRevision;
        addRevisionIds(revisionIds);

        log("Parsed " + parseFileCount + " files. " +
                "Reused " + reuseFileCount + " files. " +
                "Parsing error for " + parseErrorCount + " files. " +
                "Ignored " + filteredCount + " methods. " +
                "Total collected " + sMethodDataIdHashMap.size() + " methods.");
    }


    private static int getFileHash(Path javaFile) throws IOException {
        // Hash both the path and the content
        return Objects.hash(javaFile.toString(), Arrays.hashCode(Files.readAllBytes(javaFile)));
    }
    
    private static int addMethodData(MethodData methodData) {
        // Don't duplicate previous appeared methods (keys: path, code, comment)
        int hash = Objects.hash(methodData.path, methodData.code, methodData.comment);
        Integer prevMethodDataId = sMethodDataIdHashMap.get(hash);
        if (prevMethodDataId != null) {
            // If this method org.csevo.data already existed before, retrieve its id
            return prevMethodDataId;
        } else {
            // Allocate a new id and save this org.csevo.data to the hash map
            methodData.id = sCurrentMethodDataId;
            ++sCurrentMethodDataId;
            sMethodDataIdHashMap.put(hash, methodData.id);
            
            // Save the method org.csevo.data
            GSON.toJson(methodData, MethodData.class, sMethodDataWriter);
            return methodData.id;
        }
    }
    
    private static void addRevisionIds(RevisionIds revisionIds) {
        // Directly write to file
        GSON_NO_PPRINT.toJson(revisionIds, RevisionIds.class, sMethodProjectRevisionWriter);
    }
}
