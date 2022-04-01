package org.tseval;

import com.github.javaparser.ast.PackageDeclaration;
import com.github.javaparser.ast.body.AnnotationDeclaration;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.EnumDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.body.Parameter;
import com.github.javaparser.ast.body.TypeDeclaration;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import com.github.javaparser.javadoc.Javadoc;
import com.github.javaparser.javadoc.JavadocBlockTag;
import com.github.javaparser.javadoc.description.JavadocDescription;
import com.github.javaparser.javadoc.description.JavadocDescriptionElement;
import com.github.javaparser.javadoc.description.JavadocInlineTag;
import com.github.javaparser.printer.PrettyPrinterConfiguration;
import org.apache.commons.lang3.tuple.Pair;
import org.tseval.data.MethodData;
import org.tseval.util.NLPUtils;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

public class MethodDataCollectorVisitor extends VoidVisitorAdapter<MethodDataCollectorVisitor.Context> {
    
    private static final int METHOD_LENGTH_MAX = 10_000;
    
    public static class Context {
        String packageName = "";
        String className = "";
        List<MethodData> methodDataList = new LinkedList<>();
        Map<FilteredReason, Integer> filteredCounters = initFilteredCounters();
    }
    
    public enum FilteredReason {
        CODE_TOO_LONG("code_too_long"),
        CODE_NON_ENGLISH("code_non_english"),
        COMMENT_NON_ENGLISH("comment_non_english"),
        EMPTY_BODY("empty_body"),
        EMPTY_COMMENT_SUMMARY("empty_comment_summary"),
        EMPTY_COMMENT("empty_comment");
        
        private final String key;
        FilteredReason(String key) {
            this.key = key;
        }
    
        public String getKey() {
            return key;
        }
    }
    
    public static Map<FilteredReason, Integer> initFilteredCounters() {
        Map<FilteredReason, Integer> filteredCounters = new HashMap<>();
        for (FilteredReason fr : FilteredReason.values()) {
            filteredCounters.put(fr, 0);
        }
        return filteredCounters;
    }
    
    private static PrettyPrinterConfiguration METHOD_PPRINT_CONFIG = new PrettyPrinterConfiguration();
    static {
        METHOD_PPRINT_CONFIG
                .setPrintJavadoc(false)
                .setPrintComments(false);
    }
    
    @Override
    public void visit(ClassOrInterfaceDeclaration n, Context context) {
        commonVisitTypeDeclaration(n, context);
        super.visit(n, context);
    }
    
    @Override
    public void visit(AnnotationDeclaration n, Context context) {
        commonVisitTypeDeclaration(n, context);
        super.visit(n, context);
    }
    
    @Override
    public void visit(EnumDeclaration n, Context context) {
        commonVisitTypeDeclaration(n, context);
        super.visit(n, context);
    }
    
    public void commonVisitTypeDeclaration(TypeDeclaration<?> n, Context context) {
        // Update context class name
        context.className = n.getNameAsString();
    }
    
    @Override
    public void visit(PackageDeclaration n, Context context) {
        // Update context package name
        context.packageName = n.getNameAsString();
        super.visit(n, context);
    }
    
    private String dollaryClassName(TypeDeclaration<?> n) {
        if (n.isNestedType()) {
            return dollaryClassName((TypeDeclaration<?>) n.getParentNode().get()) + "$" + n.getNameAsString();
        }
        return n.getNameAsString();
    }
    
    @Override
    public void visit(MethodDeclaration n, Context context) {
        MethodData methodData = new MethodData();
    
        if (!n.getBody().isPresent() || n.getBody().get().isEmpty()) {
            // Ignore if no/empty method body
            context.filteredCounters.compute(FilteredReason.EMPTY_BODY, (k, v) -> v+1);
            return;
        }
    
        methodData.name = n.getNameAsString();

        if (n.getJavadoc().isPresent()) {
            Javadoc javadoc = n.getJavadoc().get();
            methodData.comment = javadoc.toText();

            if (!NLPUtils.isValidISOLatin(methodData.comment)) {
                // Ignore if comment is not English
                context.filteredCounters.compute(FilteredReason.COMMENT_NON_ENGLISH, (k, v) -> v+1);
                return;
            }
    
            methodData.commentSummary = NLPUtils.getFirstSentence(javadocDescToTextNoInlineTags(javadoc.getDescription())).orElse(null);

            if (methodData.commentSummary == null) {
                // Ignore if the comment summary is empty
                context.filteredCounters.compute(FilteredReason.EMPTY_COMMENT_SUMMARY, (k, v) -> v+1);
                return;
            }
        } else {
            // Ignore if comment is empty
            context.filteredCounters.compute(FilteredReason.EMPTY_COMMENT, (k, v) -> v+1);
            return;
        }

        methodData.code = n.toString(METHOD_PPRINT_CONFIG);
    
        // Ignore if method is too long
        if (methodData.code.length() > METHOD_LENGTH_MAX) {
            context.filteredCounters.compute(FilteredReason.CODE_TOO_LONG, (k, v) -> v+1);
            return;
        }
    
        // Ignore if method is not English
        if (!NLPUtils.isValidISOLatin(methodData.code)) {
            context.filteredCounters.compute(FilteredReason.CODE_NON_ENGLISH, (k, v) -> v+1);
            return;
        }

        MethodDeclaration maskedMD = n.clone();
        maskedMD.accept(MaskMethodNameVisitor.sVisitor, new MaskMethodNameVisitor.Context(methodData.name));
        methodData.codeMasked = maskedMD.toString(METHOD_PPRINT_CONFIG);

        try {
            methodData.cname = dollaryClassName((TypeDeclaration<?>) n.getParentNode().get());
        }
        catch (Exception e) {
            methodData.cname = context.className;
        }
        
        if (!context.packageName.isEmpty()) {
            methodData.qcname = context.packageName + "." + methodData.cname;
        } else {
            methodData.qcname = methodData.cname;
        }
    
        methodData.ret = n.getType().asString();
        for (Parameter param : n.getParameters()) {
            methodData.params.add(Pair.of(param.getType().asString(), param.getNameAsString()));
        }
        
        context.methodDataList.add(methodData);
        
        // Note: no recursive visiting the body of this method
    }
    
    static String javadocDescToTextNoInlineTags(JavadocDescription desc) {
        StringBuilder sb = new StringBuilder();
        for (JavadocDescriptionElement e : desc.getElements()) {
            if (e instanceof JavadocInlineTag) {
                sb.append(((JavadocInlineTag) e).getContent());
            } else {
                sb.append(e.toText());
            }
        }
        return sb.toString();
    }
}
