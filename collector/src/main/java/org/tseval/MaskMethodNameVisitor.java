package org.tseval;

import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.expr.SimpleName;
import com.github.javaparser.ast.visitor.ModifierVisitor;
import com.github.javaparser.ast.visitor.Visitable;

public class MaskMethodNameVisitor extends ModifierVisitor<MaskMethodNameVisitor.Context> {
    
    public static final MaskMethodNameVisitor sVisitor = new MaskMethodNameVisitor();
    
    private static final SimpleName MASK = new SimpleName("<MASK>");
    
    public static class Context {
        public String name;
        public Context(String name) {
            this.name = name;
        }
    }
    
    @Override
    public Visitable visit(MethodDeclaration n, Context arg) {
        MethodDeclaration ret = (MethodDeclaration) super.visit(n, arg);
        
        if (n.getNameAsString().equals(arg.name)) {
            ret.setName(MASK.clone());
        }
        
        return ret;
    }
    
    @Override
    public Visitable visit(MethodCallExpr n, Context arg) {
        MethodCallExpr ret = (MethodCallExpr) super.visit(n, arg);
    
        if (n.getNameAsString().equals(arg.name)) {
            ret.setName(MASK.clone());
        }
    
        return ret;
    }
}
