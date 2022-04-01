package org.tseval;

import com.github.javaparser.ParseProblemException;
import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.type.ClassOrInterfaceType;
import com.github.javaparser.printer.PrettyPrinterConfiguration;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonDeserializer;
import com.google.gson.JsonObject;
import com.google.gson.JsonParseException;
import com.google.gson.JsonSerializer;
import com.google.gson.reflect.TypeToken;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

/**
 * Utility class for extracting tokens from methods.
 * Used for tokenization and calculating metrics.
 */
public class ExtractToken {
    
    /**
     * Input data format class for {@link ExtractToken}.
     */
    public static class InputIndexAndCode {
        public int index;
        public String code;
        
        public static final JsonDeserializer<InputIndexAndCode> sDeserializer = getDeserializer();
        public static JsonDeserializer<InputIndexAndCode> getDeserializer() {
            return (json, type, context) -> {
                try {
                    InputIndexAndCode obj = new InputIndexAndCode();
                    JsonObject jObj = json.getAsJsonObject();
                    obj.index = jObj.get("index").getAsInt();
                    obj.code = jObj.get("code").getAsString();
                    return obj;
                } catch (IllegalStateException e) {
                    throw new JsonParseException(e);
                }
            };
        }
    }
    
    public static class OutputData {
        public int index;
        public List<String> tokens;
        
        public static final JsonSerializer<OutputData> sSerializer = getSerializer();
        public static JsonSerializer<OutputData> getSerializer() {
            return (d, type, jsonSerializationContext) -> {
                JsonObject jObj = new JsonObject();
                jObj.addProperty("index", d.index);
                JsonArray jTokens = new JsonArray();
                for (String token: d.tokens) {
                    jTokens.add(token);
                }
                jObj.add("tokens", jTokens);
                return jObj;
            };
        }
    }

    // Gson: For json (de)serialization
    private static final Gson GSON = new GsonBuilder()
            .disableHtmlEscaping()
            .registerTypeAdapter(InputIndexAndCode.class, InputIndexAndCode.sDeserializer)
            .registerTypeAdapter(OutputData.class, OutputData.sSerializer)
            .create();
    
    // For JavaParser pretty-printing AST to code without comments
    private static final PrettyPrinterConfiguration METHOD_PPRINT_CONFIG = new PrettyPrinterConfiguration();
    static {
        METHOD_PPRINT_CONFIG
                .setPrintJavadoc(false)
                .setPrintComments(false);
    }
    
    private static Node parseWhatever(String code) {
        // First try to parse as Compilation Unit
        try {
            return StaticJavaParser.parse(code);
        } catch (ParseProblemException ignored) {
        }
        
        // Then, try several types
        List<Class<?>> possibleTypes = Arrays.asList(
                MethodDeclaration.class,
                ClassOrInterfaceType.class
        );
        for (Class<?> t : possibleTypes) {
            try {
                Method parseMethod = StaticJavaParser.class.getMethod("parse" + t.getSimpleName(), String.class);
                Node n = (Node) parseMethod.invoke(null, code);
                return n;
            } catch (NoSuchMethodException | IllegalAccessException e) {
                throw new RuntimeException(e);
            } catch (InvocationTargetException ignored) {
            }
        }
        
        // If all fails, return null
        return null;
    }
    
    
    /**
     * Main entry point.
     *
     * @param args expect exactly two arguments: the input file path, the output file path.
     */
    public static void main(String... args) {
        // Load arguments
        if (args.length != 2) {
            throw new RuntimeException("Args: inputPath outputPath");
        }
        Path inputPath = Paths.get(args[0]);
        Path outputPath = Paths.get(args[1]);
        
        // Load inputs
        List<InputIndexAndCode> inputList;
        try (BufferedReader r = Files.newBufferedReader(inputPath)) {
            inputList = GSON.fromJson(
                    r,
                    TypeToken.getParameterized(
                            TypeToken.get(List.class).getType(),
                            TypeToken.get(InputIndexAndCode.class).getType()
                    ).getType()
            );
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        
        // Get tokens
        List<OutputData> outputList = new LinkedList<>();
        for (InputIndexAndCode input: inputList) {
            OutputData output = new OutputData();
            Node parsed = parseWhatever(input.code);
    
            if (parsed == null) {
                // Return empty list, representing the code is non-parsable
                output.index = input.index;
                output.tokens = new LinkedList<>();
            } else {
                List<String> tokens = new LinkedList<>();
                parsed.getTokenRange().get()
                        .forEach(t -> {
                            if (!t.getCategory().isWhitespaceOrComment()) {
                                tokens.add(t.getText());
                            }
                        });
                output.index = input.index;
                output.tokens = tokens;
            }

            outputList.add(output);
        }
        
        // Save outputs
        try (BufferedWriter w = Files.newBufferedWriter(outputPath)) {
            w.write(GSON.toJson(
                    outputList,
                    TypeToken.getParameterized(
                            TypeToken.get(List.class).getType(),
                            TypeToken.get(OutputData.class).getType()
                    ).getType()
            ));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
